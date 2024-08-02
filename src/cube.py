from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL.Image import open

class Cube:

    def drawcube(self, texture_id = None, texture_tile = 1.0):
        #
        # if texture_id is not None:
        #     glEnable(GL_TEXTURE_2D)
        #     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        #     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        #     # Repeat the texture.
        #     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        #     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        #
        #     glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        #
        #     glBindTexture(GL_TEXTURE_2D, texture_id)
        #
        # glBegin(GL_QUADS)
        #
        # # Textured cube.
        # glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0,  1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0,  1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0, -1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0,  1.0, -1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        # glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, -1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0, -1.0);
        # glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0,  1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0,  1.0,  1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0, -1.0, -1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0, -1.0, -1.0);
        # glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0, -1.0, -1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0,  1.0,  1.0);
        # glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        # glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);
        # glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        # glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0,  1.0,  1.0);
        # glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0, -1.0);

        # Colored cube.
        glColor3f(0.1,0.1,0.1)
        glBegin(GL_QUADS)
        szx, szy, szz = 1,1,1
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
    def drawcube2(self, texture_id = None, texture_tile = 1.0):
        #
        if texture_id is not None:
            glEnable(GL_TEXTURE_2D)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            # Repeat the texture.
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

            glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

            glBindTexture(GL_TEXTURE_2D, texture_id)

        glBegin(GL_QUADS)

        # Textured cube.
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0,  1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0,  1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0, -1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0,  1.0, -1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, -1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0, -1.0);
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0,  1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0,  1.0,  1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0, -1.0, -1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0, -1.0, -1.0);
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f( 1.0, -1.0, -1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f( 1.0,  1.0, -1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f( 1.0,  1.0,  1.0);
        glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);
        glTexCoord2f(texture_tile, 0.0); glVertex3f(-1.0, -1.0,  1.0);
        glTexCoord2f(texture_tile, texture_tile); glVertex3f(-1.0,  1.0,  1.0);
        glTexCoord2f(0.0, texture_tile); glVertex3f(-1.0,  1.0, -1.0);

        glEnd()