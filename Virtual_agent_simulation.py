# Virtual agent simulation aided by FlyVisNet

# Angel Canelo 2024.08.02

import matplotlib.pyplot as plt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from src.collision import Collision
from src.cube import Cube
from src.input import Input
from src.movement import Movement
from src.plane import Plane
from src.sprite import Sprite
from src.texture import Texture
from FlyVisNetH_regression_model import FlyVisNetH_regression
import numpy as np
import cv2
import seaborn as sns
import scipy.io
import os
import re
##############################################
def get_max_index(folder_path, file_prefix):
    max_index = 0
    for filename in os.listdir(folder_path):
        match = re.match(f"{file_prefix}_(\d+)\\.mat", filename)
        if match:
            current_index = int(match.group(1))
            max_index = max(max_index, current_index)
    return max_index
def add_next_index(folder_path, file_prefix):
    max_index = get_max_index(folder_path, file_prefix)

    # Create a new filename with the next index and ".mat" extension
    new_index = max_index + 1
    return f"{file_prefix}_{new_index}.mat"
# Example usage:
folder_path = './'
###########################################################

window = 0

# Size of cubes used to create wall segments.
cubesize = 2
# Space around cubes to extend hitbox (prevents peeking through walls).
collision_padding = 0.5
# Arena type (0: Visually guided, 1: Surveillance)
arena = 0
# Initial camera position after map is drawn.
if arena == 0:
    #camerapos = [-12, 0.0, -12.0]
    camerapos = [-12 + np.random.uniform(low=-1.0, high=1.0), 0.0, -12.0]
    # Initial camera rotation.
    camerarot = 0.0
    file_prefix = 'agent_position_trace_visually'
    new_filename = add_next_index(folder_path, file_prefix)
elif arena == 1:
    #camerapos = [-11, 0.0, -3]
    camerapos = [-11, 0.0, -3 + np.random.uniform(low=-0.5, high=0.5)]
    # Initial camera rotation.
    camerarot = 90
    file_prefix = 'agent_position_trace_surveillance'
    new_filename = add_next_index(folder_path, file_prefix)
# camerapos = [0.0, 0.0, 0.0]
# The angle in degrees the camera rotates each turn.
#rotate_angle = 0.1
rotate_angle = 0.2

# Create a VideoWriter object
record = 0  # 1 to generate video file
if record == 1:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if arena == 0:
        vid_out = cv2.VideoWriter('visually_guided_v3.mp4', fourcc, 30, (648, 244))
    elif arena == 1:
        vid_out = cv2.VideoWriter('surveillance_v3.mp4', fourcc, 30, (648, 244))
else:
    fourcc = 0
    vid_out = 0

first_run = False

collision = Collision()
input = Input();
movement = Movement();
classes_list = ['collision', 'bar', 'spot', 'non-detected']
cameraposX_list = []
cameraposY_list = []

map = []
frame_data = []

#### CNN #####
HEIGHT = 244
WIDTH = 324
classes = 3
cnn = FlyVisNetH_regression()
cnn_model = cnn.FlyVisNet_model(HEIGHT, WIDTH, classes)
cnn_model.load_weights("../WEIGHTS/FlyVisNet_regression_weights.h5")
X_out = 0
class_out = 0
### Condition variables upon detection
coll = 0
turn_l = 0
turn_r = 0
smooth = 0
smcount = 0
# Loaded textures.
ceilingtexture = None
floortexture = None
orbtexture = None
walltexture = None

def initGL(Width, Height):

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def drawScene():
    global camerapos, first_run, frame_num, frame_count, X_out, class_out, arena, vid_out, input, record

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    Width = 648
    Height = 244
    # First camera view
    glViewport(0, 0, Width // 2, Height)  # Render on left half of window
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width // 2) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Draw the scene
    draw_scene()
    glLoadIdentity()
    # Second camera view (top view)
    glViewport(Width // 2, 0, Width // 2, Height)  # Render on right half of window
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width // 2) / float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    # Draw the scene again with the second camera view
    draw_scene2()

    glutSwapBuffers()

def draw_scene():

    global camerapos, first_run, frame_num, frame_count, X_out, class_out, arena, vid_out, input, record

    cube = Cube()
    plane = Plane()
    roof = Plane()
    sprite = Sprite()

    # Set up the current maze view.
    # Reset position to zero, rotate around y-axis, restore position.
    glTranslatef(0.0, 0.0, 0.0)
    glRotatef(camerarot, 0.0, 1.0, 0.0)
    glTranslatef(camerapos[0], camerapos[1], camerapos[2])

    # Set the clear color to blue
    glClearColor(0.1, 0.8, 1.0, 0)

    # Draw floor.
    glPushMatrix()
    glTranslatef(0.0, -2.0, 0.0)
    glScalef(30.0, 1.0, 30.0)
    plane.drawplane3(floortexture, 40.0)
    glPopMatrix()

    # Draw ceiling.
    glPushMatrix()
    glTranslatef(0.0, 2.0, 0.0)
    glRotatef(180.0, 0.0, 0.0, 1.0)
    glScalef(30.0, 1.0, 30.0)
    roof.drawplane3(ceilingtexture, 50.0)
    glPopMatrix()

    # Draw shapes in the arena
    if arena == 0:
        # Square
        glPushMatrix()
        glTranslatef(12, 0, 1)
        #glRotatef(90.0, 1.0, 0.0, 0.0)
        #glRotatef(camerarot, 0.0, 0.0, 1.0)
        #glScalef(1.5, 0.0, 1.0)
        sprite.square(orbtexture)
        glPopMatrix()
        # Rectangle
        glPushMatrix()
        glTranslatef(3, 0, 5)
        sprite.rectangle(orbtexture)
        glPopMatrix()
        # Circle
        glPushMatrix()
        glTranslatef(4.5, 0, 1.1)
        sprite.circle(0.25, 100)
        glPopMatrix()
    elif arena == 1:
        # Rectangle 1
        glPushMatrix()
        glTranslatef(5, 0, 3)
        sprite.rectangle(orbtexture)
        glPopMatrix()
        # Rectangle 2
        glPushMatrix()
        glTranslatef(15, 0, 3)
        sprite.rectangle(orbtexture)
        glPopMatrix()

    # Build the maze like a printer; back to front, left to right.
    row_count = 0
    column_count = 0

    wall_x = 0.0
    wall_z = 0.0

    for i in map:

        wall_z = (row_count * (cubesize * -1))

        for j in i:

            # 1 = cube, 0 = empty space.
            if (j == 1):
                cube.drawcube2(walltexture, 1.0)

                wall_x = (column_count * (cubesize * -1))

                if (first_run != True):
                    print('Drawing cube at X:', wall_x, 'Z:', wall_z)

            # Move from left to right one cube size.
            glTranslatef(cubesize, 0.0, 0.0)

            column_count += 1

        # Reset position before starting next row, while moving
        # one cube size towards the camera.
        glTranslatef(((cubesize * column_count) * -1), 0.0, cubesize)

        row_count += 1
        # Reset the column count; this is a new row.
        column_count = 0

    #glutSwapBuffers()

    # Run FlyVisNet on each frame
    # Get the scene frame to array
    # Read the pixels from the frame buffer.
    pixels = glReadPixels(0, 0, 324, glutGet(GLUT_WINDOW_HEIGHT), GL_RGB, GL_UNSIGNED_BYTE)
    # Convert the pixels to a numpy array.
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(glutGet(GLUT_WINDOW_HEIGHT), 324, 3)
    # frame_data.append(np.flipud(image))
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    frame = np.flipud(frame)
    #plt.imshow(frame,'gray'); plt.show()
    frame = np.expand_dims(np.array(frame), axis=-1)
    frame = np.expand_dims(np.array(frame), axis=0)
    out = cnn_model(frame/255)
    X_out = np.squeeze(out[0])
    class_out = np.argmax(out[1])
    print('class =', class_out, 'X =', X_out)

    frame_count += 1
    if frame_count % 100 == 0:
        # pixels = glReadPixels(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), GL_RGB, GL_UNSIGNED_BYTE)

        # Convert the pixels to a numpy array.
        # image = np.frombuffer(pixels, dtype=np.uint8).reshape(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), 3)

        # Save the image as a PNG file.
        # filename = f"frame_{frame_num:04d}.png"
        # Image.fromarray(image).save(filename)
        # scipy.io.savemat("3Dworld_simulation_video_frames.mat", {"frames": frame_data})
        # Increment the frame number.
        frame_num += 1
    ####################################
    handleInput()

    if (first_run != True):
        first_run = True

def draw_scene2():

    global camerapos, first_run, frame_num, frame_count, X_out, class_out, arena, vid_out, input, record

    cube = Cube()
    plane = Plane()
    roof = Plane()
    sprite = Sprite()

    # Set up the current maze view.
    # Reset position to zero, rotate around y-axis, restore position.
    glTranslatef(0.0, 0.0, 0.0)
    #glRotatef(camerarot, 0.0, 1.0, 0.0)
    glRotatef(90, 1, 0, 0)
    glTranslatef(camerapos[0], camerapos[1]-20, camerapos[2])

    # Set the clear color to blue
    glClearColor(0.1, 0.8, 1.0, 0)

    # Draw floor.
    glPushMatrix()
    glTranslatef(0.0, -2.0, 0.0)
    glScalef(30.0, 1.0, 30.0)
    plane.drawplane3(floortexture, 40.0)
    glPopMatrix()

    # Draw ceiling.
    # glPushMatrix()
    # glTranslatef(0.0, 2.0, 0.0)
    # glRotatef(180.0, 0.0, 0.0, 1.0)
    # glScalef(30.0, 1.0, 30.0)
    # roof.drawplane3(ceilingtexture, 50.0)
    # glPopMatrix()

    # Draw shapes in the arena
    if arena == 0:
        # Square
        glPushMatrix()
        glTranslatef(12, 0, 1)
        #glRotatef(90.0, 1.0, 0.0, 0.0)
        #glRotatef(camerarot, 0.0, 0.0, 1.0)
        #glScalef(1.5, 0.0, 1.0)
        sprite.square(orbtexture)
        glPopMatrix()
        # Rectangle
        glPushMatrix()
        glTranslatef(3, 0, 5)
        sprite.rectangle(orbtexture)
        glPopMatrix()
        # Circle
        glPushMatrix()
        glTranslatef(4.5, 0, 1.1)
        sprite.circle(0.25, 100)
        glPopMatrix()
    elif arena == 1:
        # Rectangle 1
        glPushMatrix()
        glTranslatef(5, 0, 3)
        sprite.rectangle(orbtexture)
        glPopMatrix()
        # Rectangle 2
        glPushMatrix()
        glTranslatef(15, 0, 3)
        sprite.rectangle(orbtexture)
        glPopMatrix()

    # Build the maze like a printer; back to front, left to right.
    row_count = 0
    column_count = 0

    wall_x = 0.0
    wall_z = 0.0

    for i in map:

        wall_z = (row_count * (cubesize * -1))

        for j in i:

            # 1 = cube, 0 = empty space.
            if (j == 1):
                cube.drawcube2(walltexture, 1.0)

                wall_x = (column_count * (cubesize * -1))

                if (first_run != True):
                    print('Drawing cube at X:', wall_x, 'Z:', wall_z)

            # Move from left to right one cube size.
            glTranslatef(cubesize, 0.0, 0.0)

            column_count += 1

        # Reset position before starting next row, while moving
        # one cube size towards the camera.
        glTranslatef(((cubesize * column_count) * -1), 0.0, cubesize)

        row_count += 1
        # Reset the column count; this is a new row.
        column_count = 0
    if arena == 0:
        cameraposX_list.append(-camerapos[0]-12)
        cameraposY_list.append(camerapos[2]+12)
    elif arena == 1:
        cameraposX_list.append(-camerapos[0]-11)
        cameraposY_list.append(camerapos[2]+3)
    #### Write frame to video file #########
    if record == 1:
        pixels2 = glReadPixels(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), GL_RGB, GL_UNSIGNED_BYTE)
        image2 = np.frombuffer(pixels2, dtype=np.uint8).reshape(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), 3)
        snapshot = image2[:,:,::-1]
        snapshot = cv2.flip(snapshot, 0)
        #text = f'class = {class_out:d}, X = {int(X_out):d}'
        if arena == 0:
            text = f'class = {classes_list[class_out]}, X = {int(X_out):d}'
        elif arena == 1:
            if class_out == 0:
                text = f'class = {classes_list[class_out]}, X = {int(X_out):d}'
            else:
                text = f'class = non-collision, X = {int(X_out):d}'
        # Get text size and position
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2)
        if arena == 0:
            text_x = snapshot.shape[0] - text_size[0] + 14
            text_y = text_size[1] + 210
        elif arena == 1:
            text_x = snapshot.shape[0] - text_size[0] + 70
            text_y = text_size[1] + 210
        cv2.putText(snapshot, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), thickness=2)
        cv2.putText(snapshot, '+', (int(snapshot.shape[1]-snapshot.shape[1]/4), int(244/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)
        cv2.line(snapshot, (324, 0), (324, 244), (255, 255, 255), thickness=2)
        vid_out.write(snapshot) # write frame to video
        ###############################################
        # if input.isKeyDown(Input.KEY_STATE_ESCAPE):
        #     vid_out.release()
    ##################################
    #glutSwapBuffers()

def handleInput():

    global input, camerapos, camerarot, X_out, class_out, coll, turn_l, turn_r, arena, vid_out, smooth, smcount

    if input.isKeyDown(Input.KEY_STATE_ESCAPE):
        vid_out.release()
        sys.exit()
    if input.isKeyDown(Input.KEY_STATE_FORWARD) and arena == 0:
        sns.set()
        # scipy.io.savemat(new_filename, {"X": cameraposX_list, "Y": cameraposY_list})
        plt.plot(cameraposX_list, cameraposY_list)
        plt.xlabel('X (a.u.)')
        plt.ylabel('Y (a.u.)')
        plt.show()
        sys.exit()
    elif input.isKeyDown(Input.KEY_STATE_FORWARD) and arena == 1:
        sns.set()
        # scipy.io.savemat(new_filename, {"X": cameraposX_list, "Y": cameraposY_list})
        plt.plot(cameraposX_list, cameraposY_list)
        plt.xlabel('X (a.u.)')
        plt.ylabel('Y (a.u.)')
        plt.ylim(-0.5, 0.5)
        plt.show()
        sys.exit()


    ## Keep object in the center of the visual field ##
    if X_out < 152:
        camerarot -= rotate_angle

    if X_out > 172:
        camerarot += rotate_angle

    intended_pos = [camerapos[0], 0, camerapos[2]];
    ## Check infered class to turn 90 deg left/right ##
    if arena == 0:
        if smooth == 1 and smcount <= 90 and turn_l == 1 and turn_r == 0:
            camerarot -= 2
            smcount += 2
        elif smooth == 1 and smcount <= 90 and turn_r == 1:
                camerarot += 2
                smcount += 2
        else:
            if class_out != 0 and coll < 2: #coll < 4:     # Go straight
                modifier = 1
                intended_pos = movement.getIntendedPosition(camerarot, camerapos[0], camerapos[2], 90, modifier)
                smooth = 0
            elif class_out == 0 and turn_r == 0:    # Detect collision to square
                modifier = -1
                intended_pos = movement.getIntendedPosition(camerarot, camerapos[0], camerapos[2], 90, modifier)
                coll += 1
                smooth = 0
            elif coll > 1 and turn_l == 1 and turn_r == 0:   # Collision to rectangle, then turn right
                #camerarot += 90
                coll = 0
                turn_r = 1
                smooth = 1
                smcount = 0
            elif coll > 1 and turn_l == 0:      # Collision to square, then turn left
                #camerarot -= 90
                coll = 0
                turn_l = 1
                smooth = 1
                smcount = 0
        ## After collision detection stop moving ##
            if coll > 0 and turn_r == 1:      # class_out == 0
                intended_pos = movement.getIntendedPosition(0, camerapos[0], camerapos[2], 90, 0)
                coll += 1
                print('The end')
                #sys.exit()
    elif arena == 1:
        if smooth == 1 and smcount <= 180:
            camerarot -= 2
            smcount += 2
        else:
            if class_out != 0 and coll == 0:# and camerapos[2] < -5:
                modifier = 1
                intended_pos = movement.getIntendedPosition(camerarot, camerapos[0], camerapos[2], 90, modifier)
                smooth = 0
            elif class_out == 0:
                modifier = -1
                intended_pos = movement.getIntendedPosition(camerarot, camerapos[0], camerapos[2], 90, modifier)
                coll += 1
                smooth = 0
            elif coll >= 1:
                #camerarot += 180
                coll = 0
                smooth = 1
                smcount = 0

    intended_x = intended_pos[0]
    intended_z = intended_pos[2]

    # Move camera if there are no walls in the way.
    if (collision.testCollision(cubesize, map, intended_x, intended_z, collision_padding)):
        print('Collision at X:', intended_x, 'Z:', intended_z)
    else:
        camerapos[0] = intended_x
        camerapos[2] = intended_z

def main():

    global window, ceilingtexture, floortexture, orbtexture, walltexture, map, frame_num, frame_count, arena

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    #glutInitWindowSize(640, 480)
    glutInitWindowSize(648, 244)
    glutInitWindowPosition(200, 200)

    window = glutCreateWindow('Experimental Maze')

    # Generate map.
    # generator = Generator()
    # map = generator.generateMap(16)

    # Represents a top-down view of the maze.
    if arena == 0:
    ## Visually-guided arena ##
        map = [
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    elif arena == 1:
    ## Surveillance arena ##
        map = [
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        ]

    # Load texture.
    texture = Texture()

    #ceilingtexture = texture.loadImage('tex/ceiling.bmp')
    ceilingtexture = texture.loadImage('tex/ceiling_tile_noborder_1.png')
    #floortexture = texture.loadImage('tex/floor.bmp')
    floortexture = texture.loadImage('tex/concrete_tiles_1.png')
    orbtexture = texture.loadImage('tex/orb.bmp')
    #walltexture = texture.loadImage('tex/wall.bmp')
    walltexture = texture.loadImage('tex/grungeconcrete2.jpg')

    glutIgnoreKeyRepeat(1)
    glutKeyboardFunc(input.registerKeyDown)
    glutKeyboardUpFunc(input.registerKeyUp)

    glutDisplayFunc(drawScene)
    glutIdleFunc(drawScene)
    initGL(648, 244)
    frame_num = 0
    frame_count = 0
    glutMainLoop()

if __name__ == "__main__":

    main()
