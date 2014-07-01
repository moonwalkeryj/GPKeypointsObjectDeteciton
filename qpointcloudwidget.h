#ifndef QPOINTCLOUDWIDGET_H
#define QPOINTCLOUDWIDGET_H

#include <QVTKWidget.h>

#ifdef __GNUC__
#  pragma GCC system_header
#endif

#ifndef Q_MOC_RUN
#include <boost/static_assert.hpp>
#include <boost/unordered_map.hpp>
#include <boost/bind/bind.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/signals2/connection.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <boost/type_traits/is_same.hpp>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif
#include <opencv2/opencv.hpp>
#include <QTimer>

namespace pcl {
	namespace visualization {
		class PCLVisualizer;
	}
}

class QPointCloudWidget : public QVTKWidget
{
    Q_OBJECT
private:
    typedef pcl::PointXYZRGB               PointT;
    typedef pcl::PointCloud <PointT>        CloudT;
    typedef CloudT::Ptr                     CloudTPtr;
    typedef CloudT::ConstPtr                CloudTConstPtr;

    typedef pcl::PointXYZRGBNormal              PointXYZRGBNormal;
    typedef pcl::PointCloud <PointXYZRGBNormal> CloudXYZRGBNormal;
    typedef CloudXYZRGBNormal::Ptr              CloudXYZRGBNormalPtr;
    typedef CloudXYZRGBNormal::ConstPtr         CloudXYZRGBNormalConstPtr;


public:
    QPointCloudWidget (QWidget *parent);
    ~QPointCloudWidget ();
    pcl::visualization::PCLVisualizer * getPCLVisualizer();
    void showPointCloud(const CloudTConstPtr & cloud);
   // void showPointCloud(const pcl::PointCloud<pcl::PointXYZ >::ConstPtr & cloud);
	
	void 	mousePressEvent (QMouseEvent *event);
    void 	mouseMoveEvent (QMouseEvent *event);
    void 	mouseReleaseEvent (QMouseEvent *event);
    void 	wheelEvent (QWheelEvent *event);
    void    paintEvent (QPaintEvent *event);
    void    resizeEvent(QResizeEvent *event);

    void    viewupdate();
public slots:
    void resetCamera();
    void setCameraParameters(int pos, double value);

public:
    enum CAMERAPOSPARAS {POSX = 0, POSY, POSZ, VIEWX, VIEWY, VIEWZ, UPX, UPY, UPZ};
    double cam_PosParas[9];
    /** \brief Synchronization. */
    boost::mutex mutex_vis_;

private:
    CloudTConstPtr cloud;
    // is new grabbed cloud available
    bool new_cloud_available;
    QTimer *timer;
    pcl::visualization::PCLVisualizer *pviz;
};

#endif
