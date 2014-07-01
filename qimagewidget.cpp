#include "qimagewidget.h"


QImageWidget::QImageWidget(QWidget *parent) :
   QWidget(parent)
{
    cv::Mat zeros = cv::Mat::zeros(10,10,CV_8UC3);
    mQImage = QImage((unsigned char *)zeros.data,
                    zeros.cols, zeros.rows,
                    zeros.step,
                    QImage::Format_RGB888).rgbSwapped();

   setMouseTracking(true);
}

void QImageWidget::imshow(cv::Mat & image){
   //mFrameToDisplay = image.clone();
   mQImage = QImage((unsigned char *)image.data,
                   image.cols, image.rows,
                   image.step, QImage::Format_RGB888).rgbSwapped();
   update();
}

void QImageWidget::imshow(std::string s){
   mFrameToDisplay = cv::imread(s);
   mQImage = QImage((unsigned char *)mFrameToDisplay.data,
                   mFrameToDisplay.cols, mFrameToDisplay.rows,
                   mFrameToDisplay.step,
                   QImage::Format_RGB888).rgbSwapped();
   //qImage.scaled(this->size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
   update();
}

void QImageWidget::paintEvent(QPaintEvent *){
   QPainter painter(this);
   //painter.drawImage(0 , 0, qImage);
//   painter.drawImage(QRect(0, 0, this->width(), this->height()),
//                     mQImage);
    painter.drawImage(QRect(0, 0, mQImage.width(), mQImage.height()),
                    mQImage);
}

void QImageWidget::mouseMoveEvent(QMouseEvent *e){
   QString toolTip = tr("DWPos:");
   toolTip.append(QString::number(e->pos().x())).append(tr(",")).append(QString::number( e->pos().y()));

   QRect rect(0,0,50,20);
   QToolTip::showText(e->globalPos(),toolTip, this, rect);
}

void QImageWidget::mousePressEvent(QMouseEvent *e)
{
    emit pointClicked(e->pos().x(), e->pos().y());
}
