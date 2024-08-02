# FlyVisNet
This repository contains the essential code used to generate the primary results and conduct the simulations in: "FlyVisNet – An insect vision-inspired artificial neural network for the control of a nano drone".

The first part of the repository describes the files necessary to generate the primary results of the study. The second part explains the deployment process for embedding FlyVisNet in the Crazyflie drone.

## Training and simulations
The order to execute the files and their details are as follows:
- **Generate_moving_patterns.py** generates the moving patterns dataset necessary to train and test the models.
- **FlyVisNetH_model.py** contains the model for the resolution of input images of 244x324.
- **FlyVisNetL_model.py** contains the model for the resolution of input images of 20x40.
- **Train_models.py** allows to train FlyVisNetH, FlyVisNetL, Dronet, and MobileNet using the pattern dataset, and COIL-100 dataset.
- **plot_accuracy_performance.py** generates the graphs for accuracy performance comparison between the different architectures.
- **FlyVisNet_activity.py** generates the graphs for the convolution kernels, 2D feature maps, and 1D activations.
- **FlyVisNetH_pruning_model.py** contains the model that allows pruning each layer separately.
- **Train_pruning_models.py** trains the FlyVisNetH model pruning one convolution layer per time.
- 

## Deployment
The necessary components for deployment are as follow:
- Crazyflie 2.1 drone
- Crazyradio PA 2.4 GHz USB dongle
- Flow deck v2
- AI deck 1.1

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/necessary_components.jpg" width=40% height=40%>

Instructions for deployment on *crazyflie 2.1* and *ai-deck*:
- Download *bitcraze-vm* https://github.com/bitcraze/bitcraze-vm/releases
- On the vm clone *aideck-gap8-examples*, and *crazyflie-firmware* repositories: <br/>
https://github.com/bitcraze/aideck-gap8-examples <br/>
https://github.com/bitcraze/crazyflie-firmware
- Substitute the folder *classification* in `aideck-gap8-examples/examples/ai/` by the provided by us in `deployment/classification`
- Substitute the folder *app_hello_world* in `crazyflie-firmware/examples/` by the provided by us in `deployment/app_hello_world`

- Build and flash on *ai-deck* GAP8. In folder `aideck-gap8-examples`:
```
$ docker run --rm -v ${PWD}:/module aideck-with-autotiler tools/build/make-example examples/ai/classification clean model build image
```
```
$ cfloader flash examples/ai/classification/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img deck-bcAI:gap8-fw -w radio://0/80/2M/E7E7E7E7E7
```
- Build and flash on *crazyflie* STM32. In folder `crazyflie-firmware/examples/app_hello_world`:
```
$ make all clean
```
```
$ cfloader flash ./build/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7
```
