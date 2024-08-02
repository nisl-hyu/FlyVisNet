from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL.Image import open
import math

class Sprite:

    def square(self, texture_id = None):

        glColor3f(1,1,1)
        glBegin(GL_QUADS)
        szx, szy, szz = 0.25,0.25,0.25
        # Draw front face
        glVertex3f(szx, szy, szz)
        glVertex3f(-szx, szy, szz)
        glVertex3f(-szx, -szy, szz)
        glVertex3f(szx, -szy, szz)

        # Draw back face
        glVertex3f(szx, -szy, -szz)
        glVertex3f(-szx, -szy, -szz)
        glVertex3f(-szx, szy, -szz)
        glVertex3f(szx, szy, -szz)

        # Draw left face
        glVertex3f(-szx, szy, szz)
        glVertex3f(-szx, szy, -szz)
        glVertex3f(-szx, -szy, -szz)
        glVertex3f(-szx, -szy, szz)

        # Draw right face
        glVertex3f(szx, szy, -szz)
        glVertex3f(szx, szy, szz)
        glVertex3f(szx, -szy, szz)
        glVertex3f(szx, -szy, -szz)

        # Draw top face
        glVertex3f(-szx, szy, szz)
        glVertex3f(szx, szy, szz)
        glVertex3f(szx, szy, -szz)
        glVertex3f(-szx, szy, -szz)

        # Draw bottom face
        glVertex3f(-szx, -szy, szz)
        glVertex3f(szx, -szy, szz)
        glVertex3f(szx, -szy, -szz)
        glVertex3f(-szx, -szy, -szz)

        glEnd()

    def rectangle(self, texture_id=None):
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        szx, szy, szz = 0.25, 0.75, 0.25
        # Draw front face
        glVertex3f(szx, szy, szz)
        glVertex3f(-szx, szy, szz)
        glVertex3f(-szx, -szy, szz)
        glVertex3f(szx, -szy, szz)

        # Draw back face
        glVertex3f(szx, -szy, -szz)
        glVertex3f(-szx, -szy, -szz)
        glVertex3f(-szx, szy, -szz)
        glVertex3f(szx, szy, -szz)

        # Draw left face
        glVertex3f(-szx, szy, szz)
        glVertex3f(-szx, szy, -szz)
        glVertex3f(-szx, -szy, -szz)
        glVertex3f(-szx, -szy, szz)

        # Draw right face
        glVertex3f(szx, szy, -szz)
        glVertex3f(szx, szy, szz)
        glVertex3f(szx, -szy, szz)
        glVertex3f(szx, -szy, -szz)

        # Draw top face
        glVertex3f(-szx, szy, szz)
        glVertex3f(szx, szy, szz)
        glVertex3f(szx, szy, -szz)
        glVertex3f(-szx, szy, -szz)

        # Draw bottom face
        glVertex3f(-szx, -szy, szz)
        glVertex3f(szx, -szy, szz)
        glVertex3f(szx, -szy, -szz)
        glVertex3f(-szx, -szy, -szz)

        glEnd()

    def circle(self, radius, num_segments):
        glColor3f(1, 1, 1)
        angle = 2 * math.pi / num_segments
        glBegin(GL_POLYGON)
        for i in range(num_segments):
            x = radius * math.cos(i * angle)
            y = radius * math.sin(i * angle)
            z = 0
            glVertex3f(x, y, z)

        glEnd()