#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <gl/GLU.h>

class GLWidget : public QGLWidget
{
    Q_OBJECT
public:
    explicit GLWidget(QWidget *parent = 0);
    
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    
};

#endif // GLWIDGET_H
