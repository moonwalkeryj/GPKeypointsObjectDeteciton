#include "glwidget.h"

GLWidget::GLWidget(QWidget *parent) :
    QGLWidget(parent)
{
}

void GLWidget::initializeGL(){
    glClearColor(0,0,0,0);
}
void GLWidget::paintGL(){
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    for(int r = 1; r < 250; ++r)
    {
        glVertex3i(2, r, r);
        //glVertex(2, r, r);
    }
    glEnd();

}
void GLWidget::resizeGL(int width, int height){
    if ( height == 0 ){
        height = 1;
    }
    glViewport( 0, 0, (GLint)width, (GLint)height );
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( 45.0, (GLfloat)width/(GLfloat)height, 0.1, 100.0 );
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}
