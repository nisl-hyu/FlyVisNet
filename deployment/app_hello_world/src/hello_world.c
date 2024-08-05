/*-----------------------------------------------------------------------------
 Crazyflie drone STM32 deployment of algorithm for autonomous flight.
 The FlyVisNet inference from AI-deck is received via UART.
 Code to build and flash on STM32

 Angel Canelo 2024.08.02

 Code modified from Bitcraze crazyflie-firmware app_hello_world                                                  
-------------------------------------------------------------------------------*/

#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "FreeRTOS.h"
#include "task.h"
#include "debug.h"
#include "uart_dma_setup.h"
#include "commander.h"
#include "log.h"
#include "param.h"
#include "classificationInfo.h"	// for the dequantize scale

#define DEBUG_MODULE "HELLOWORLD"
#define BUFFERSIZE 2

uint8_t aideckRxBuffer[BUFFERSIZE];
volatile uint8_t dma_flag = 0;
uint8_t log_counter=0;
int result[10];	// array of 5 elements
int elem;
float chtime;
float chtime2;

static void setHoverSetpoint(setpoint_t *setpoint, float vx, float vy, float z, float yawrate)
{
  setpoint->mode.z = modeAbs;
  setpoint->position.z = z;


  setpoint->mode.yaw = modeVelocity;
  setpoint->attitudeRate.yaw = yawrate;

  setpoint->mode.x = modeVelocity;
  setpoint->mode.y = modeVelocity;
  setpoint->velocity.x = vx;
  setpoint->velocity.y = vy;

  setpoint->velocity_body = true;
}

typedef enum {
    idle,
    //lowUnlock,
    unlocked,
    stopping
} State;

static State state = idle;

static const float height_sp = 0.2f;
static float height_sp2 = 0.3f;

#define MAX(a,b) ((a>b)?a:b)
#define MIN(a,b) ((a<b)?a:b)

int get_most_common(int vet[], size_t dim)
{
    size_t i, j, count;
    size_t most = 0;
    int temp;

    for(i = 0; i < dim; i++) {
        temp = vet[i];
        count = 1;
        for(j = i + 1; j < dim; j++) {
            if(vet[j] == temp) {
                count++;
            }
        }
        if (most < count) {
            most = count;
            elem = vet[i];
        }
    }
    return elem;
}

void appMain()
{

	static setpoint_t setpoint;

  	vTaskDelay(M2T(10000));
  	paramVarId_t idPositioningDeck = paramGetVarId("deck", "bcFlow2");

	USART_DMA_Start(115200, aideckRxBuffer, BUFFERSIZE);
	int count = 0;
	int finres = 3;
	int ddcheck = 0;
    float scaled = 0;
	uint16_t scaled16 = 0;
	while(1) {
		vTaskDelay(M2T(10));
		uint8_t positioningInit = paramGetUint(idPositioningDeck);
		if (state == unlocked)
		{
			if (finres==3 && usecTimestamp()/1000000<=chtime2+2) {	// Initial hovering
				setHoverSetpoint(&setpoint, 0, 0, height_sp, 0);
				commanderSetSetpoint(&setpoint, 3);				
			}
			else if (finres==3 && usecTimestamp()/1000000>chtime2+2){
				setHoverSetpoint(&setpoint, 0, 0, height_sp2, 0);
				commanderSetSetpoint(&setpoint, 3);
				if (usecTimestamp()/1000000>=chtime2+8){
					finres = 4;
                    chtime4 = usecTimestamp()/1000000;
				}
			}
			else {
				if (ddcheck==1){
					setHoverSetpoint(&setpoint, 0.1f, 0, height_sp2, 0);
					commanderSetSetpoint(&setpoint, 3);
					finres = 5;
					if (usecTimestamp()/1000000>chtime+2){
						finres = 0;
						state = stopping;
					}
				}
				if (finres==4 && usecTimestamp()/1000000<=chtime4+6.0f) {	// Moving fordward
					setHoverSetpoint(&setpoint, 0.1f, 0, height_sp2, 0);
					commanderSetSetpoint(&setpoint, 3);
					if (dma_flag == 1) {
						dma_flag = 0;  // clear the flag
						result[count] = aideckRxBuffer[0];
                        //////////////////
                        scaled = aideckRxBuffer[1]*classification_Output_1_OUT_SCALE;	// dequantizing
                        if (scaled >= 0 && scaled <= UINT16_MAX){
                            scaled16 = (uint16_t)scaled;
                        }
                        else if (scaled > UINT16_MAX){
                            scaled16 = UINT16_MAX;
                        }
                        else {
                            scaled16 = 0;
                        }
                        ///////////////////
                        /// Check class
                        ///////////////////
						if (count==5){
							finres = get_most_common(result, 5);
							count = 0;
							if (finres==0)
							{
								DEBUG_PRINT("Collision: %d\n", finres);
							}
							else if (finres==1)
							{
								DEBUG_PRINT("Rectangle: %d\n", finres);
                                finres = 4;
							}
							else if (finres==2)
							{
								DEBUG_PRINT("Square: %d\n", finres);
                                finres = 4;
							}
						}
						log_counter = aideckRxBuffer[0];
						memset(aideckRxBuffer, 0, BUFFERSIZE);  // clear the dma buffer
						count = count + 1;
					}
                    /// Check X coordinate
                    if (scaled < 80) {
                        setHoverSetpoint(&setpoint, 0.1f, 0, height_sp2, 1.0f);    // object to the left
                        commanderSetSetpoint(&setpoint, 3);
                    }
                    if (scaled > 140) {
                        setHoverSetpoint(&setpoint, 0.1f, 0, height_sp2, -1.0f);     // object to the right
                        commanderSetSetpoint(&setpoint, 3);
                    }
                    ///////////////////
				else if (finres==4 && turn_l == 0 && usecTimestamp()/1000000>chtime4+6.0f){
					finres = 2;
					chtime = usecTimestamp()/1000000;
				}
                else if (finres==4 && turn_l == 1 && turn_r == 0 && usecTimestamp()/1000000>chtime+6.0f){
					finres = 1;
					chtime = usecTimestamp()/1000000;
				}
                else if (turn_l == 1 && turn_r == 1 && usecTimestamp()/1000000>chtime+2.0f){
					state = stopping;
					chtime = usecTimestamp()/1000000;
				}
				}
				if (finres==0) {
					setHoverSetpoint(&setpoint, 0, 0, 0.1f, 0);
					commanderSetSetpoint(&setpoint, 3);
					state = stopping;
				}
				if (finres==1 && usecTimestamp()/1000000<=chtime+2.5f) {	// 3 sec rotation of 30deg/s (90 deg)
					setHoverSetpoint(&setpoint, 0, 0, height_sp2, -30.0f);
					commanderSetSetpoint(&setpoint, 3);
				}
				else if (finres==1 && usecTimestamp()/1000000>chtime+2.5f){
					finres = 4;
					chtime = usecTimestamp()/1000000;
                    turn_r = 1;
				}
				if (finres==2 && usecTimestamp()/1000000<=chtime+2.5f) {	// 3 sec rotation of -30deg/s (-90 deg)
					setHoverSetpoint(&setpoint, 0, 0, height_sp2, 30.0f);
					commanderSetSetpoint(&setpoint, 3);
				}
				else if (finres==2 && usecTimestamp()/1000000>chtime+2.5f){
					finres = 4;
                    chtime = usecTimestamp()/1000000;
                    turn_l = 1;
				}			
			}
		}
		///// Check state conditions
		else{

			if (positioningInit && state == idle)
			{
				DEBUG_PRINT("Unlocked!\n");
				state = unlocked;
				chtime2 = usecTimestamp()/1000000;
			}
			if (state == stopping)
			{
				memset(&setpoint, 0, sizeof(setpoint_t));
				commanderSetSetpoint(&setpoint, 3);
				DEBUG_PRINT("Collision: %d\n", finres);
				DEBUG_PRINT("Landed\n");
			}
		}
	}
}


void __attribute__((used)) DMA1_Stream1_IRQHandler(void)
{
 DMA_ClearFlag(DMA1_Stream1, UART3_RX_DMA_ALL_FLAGS);
 dma_flag = 1;
}

LOG_GROUP_START(log_test)
LOG_ADD(LOG_UINT8, test_variable_x, &log_counter)
LOG_GROUP_STOP(log_test)
