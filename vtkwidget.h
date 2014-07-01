#ifndef VTKWIDGET_H
#define VTKWIDGET_H

#include <QVTKWidget.h>

#ifdef __GNUC__
#  pragma GCC system_header
#endif

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
#include <pcl/geometry/triangle_mesh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <QTimer>
#include "cloudprocessor.hpp"
#include "mainwindow.h"
namespace pcl
{
class OpenNIGrabber;
} // End namespace pcl


namespace Ui {
class MainWindow;
}
namespace pcl {
namespace visualization {
class PCLVisualizer;
}
}

class VTKWidget : public QVTKWidget
{
    Q_OBJECT

private:
    typedef pcl::PointXYZRGBA               PointT;
    typedef pcl::PointCloud <PointT>        CloudT;
    typedef CloudT::Ptr                     CloudTPtr;
    typedef CloudT::ConstPtr                CloudTConstPtr;

    typedef pcl::PointXYZRGBNormal              PointXYZRGBNormal;
    typedef pcl::PointCloud <PointXYZRGBNormal> CloudXYZRGBNormal;
    typedef CloudXYZRGBNormal::Ptr              CloudXYZRGBNormalPtr;
    typedef CloudXYZRGBNormal::ConstPtr         CloudXYZRGBNormalConstPtr;

    typedef pcl::OpenNIGrabber                Grabber;
    typedef boost::shared_ptr <Grabber>       GrabberPtr;
    typedef boost::shared_ptr <const Grabber> GrabberConstPtr;

public:
    class StopWatch{
    private:
        double old;
    public:
        StopWatch():old(0.0){
        }
        inline void initial(){old = pcl::getTime();}
        // show computation time and recount
        inline void show(std::string title){
            cerr << "StopWatch-Computation Time of "<< title << " : " << pcl::getTime() - old << "s" << endl;
            old = pcl::getTime();
        }
    };

public:
    VTKWidget (QWidget *parent);
    ~VTKWidget ();
    void 	mousePressEvent (QMouseEvent *event);
    void 	mouseMoveEvent (QMouseEvent *event);
    void 	mouseReleaseEvent (QMouseEvent *event);
    void 	wheelEvent (QWheelEvent *event);
    void    paintEvent (QPaintEvent *event);
    void    resizeEvent(QResizeEvent *event);

    bool loadPCD(std::string str, CloudT &cloudXYZRGBA);
    pcl::visualization::PCLVisualizer * getPCLVisualizer();


public slots:
    /** \brief Start the grabber (enables the scanning pipeline). */
    void startGrabber ();
    void updateCloud();
    void resetCamera();
    void saveCurrentPointCloud();
    bool openPCDFile(std::string str);
    bool startKinectGrabber();
    bool closeKinectGrabber();
    void setCameraParameters(int pos, double value);
    void updateOpenCloud();

    void adjustPassThroughValueXMax(int value);
    void adjustPassThroughValueYMax(int value);
    void adjustPassThroughValueZMax(int value);

    void adjustPassThroughValueXMin(int value);
    void adjustPassThroughValueYMin(int value);
    void adjustPassThroughValueZMin(int value);

    // slot setter for isshow pass through filtered cloud
    void setIsShowPTFilteredCloud(bool isShowPTFilteredCloud);
    // slot setter for is show object distance
    void setIsShowObjectDistance(bool isShowObjectDistance);
    // slot setter for plane model extract distance <default value 0.03>
    void setPlaneModelSelectDistance(double planeModelSelectDistance);

signals:
    void newCloudAvailable();
    void openCloudAvailable();

private:
    /** \brief Actual implementeation of startGrabber (needed so it can be run in a different thread and doesn't block the application when starting up). */
    void startGrabberImpl ();
    /** \brief Called when new data arries from the grabber. The grabbing - registration - integration pipeline is implemented here. */
    void newDataCallbackXYZRGBA (const CloudTConstPtr &cloud_in);

public:
    enum CAMERAPOSPARAS {POSX = 0, POSY, POSZ, VIEWX, VIEWY, VIEWZ, UPX, UPY, UPZ};
    double cam_PosParas[9];
    /** \brief Synchronization. */
    boost::mutex mutex_;
    /** \brief Synchronization. */
    boost::mutex mutex_vis_;

    // getter of is show Pass Through Filtered cloud
    bool isShowPTFilteredCloud() const;
    // getter of plane model select distance threshold
    float planeModelSelectDistance() const;

    void setMainWindow(MainWindow *mainWindow);

    bool isShowObjectDistance() const;


private:
    CloudTConstPtr cloud;

    // Cloud Processor
    boost::shared_ptr<CloudProcessor<PointT>> _processor;

    // Pass through filter
    pcl::PassThrough<PointT>::Ptr _passThrough;

    // Pass through parameters
    enum PASSTHROUGHPARAS {XMIN = 0, XMAX, YMIN, YMAX, ZMIN, ZMAX};
    float _passThroughParas[6];

    // Whether show passthrough filtered data
    bool _isShowPTFilteredCloud;
    // whether show object distance
    bool _isShowObjectDistance;


    // Plane Model select within distance size
    float _planeModelSelectDistance;
    // Plane Model Coefficients
    Eigen::VectorXf _planeModelCoeff;
    // is new grabbed cloud available
    bool new_cloud_available;

    QTimer *timer;

    pcl::visualization::PCLVisualizer *pviz;
    /** \brief Used to get new data from the sensor. */
    GrabberPtr grabber_;
    /** \brief This variable is true if the grabber is starting. */
    bool starting_grabber_;
    /** \brief Connection of the grabber signal with the data processing thread. */
    boost::signals2::connection new_data_connection_;
    boost::signals2::connection new_image_connection_;
    /** \brief Prevent the application to crash while closing. */
    bool destructor_called_;

    StopWatch _stopWatch;

    MainWindow * _mainWindow;

};

#endif // VTKWIDGET_H
