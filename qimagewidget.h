#ifndef QIMAGEWIDGET_H
#define QIMAGEWIDGET_H

#include <QWidget>
#include <QTimer>
#include <QImage>
#include <QPainter>
#include <QMouseEvent>
#include <QToolTip>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class QImageWidget : public QWidget
{
   Q_OBJECT
public:
   explicit QImageWidget(QWidget *parent = 0);
   void imshow(cv::Mat &);
   void imshow(std::string);

protected:
   void paintEvent (QPaintEvent *);
   void mouseMoveEvent(QMouseEvent *);
   void mousePressEvent(QMouseEvent *e);

signals:
   void pointClicked(int x, int y);

protected:
   QImage mQImage;
   cv::Mat mFrameToDisplay;

};

#endif
