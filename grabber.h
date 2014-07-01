#ifndef GRABBER_H
#define GRABBER_H

#include<QObject>

#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>

class Controller;

class Grabber {

private:
    typedef pcl::PointXYZRGBA               PointT;
    typedef pcl::PointCloud <PointT>        CloudT;
    typedef CloudT::Ptr                     CloudTPtr;
    typedef CloudT::ConstPtr                CloudTConstPtr;

    typedef pcl::OpenNIGrabber                CPGrabber;
    typedef boost::shared_ptr <CPGrabber>       CPGrabberPtr;
    typedef boost::shared_ptr <const CPGrabber> CPGrabberConstPtr;


public:
	Grabber();
    Grabber(QObject * controller);
	~Grabber();
    void startGrabber();
    void closeGrabber();
    void startGrabberImpl ();

private:
    /** \brief Used to get new data from the sensor. */
    CPGrabberPtr grabber_;
    /** \brief This variable is true if the grabber is starting. */
    bool starting_grabber_;
    /** \brief Connection of the grabber signal with the data processing thread. */
    boost::signals2::connection new_data_connection_;

    Controller * controller_;
};


#endif // GRABBER_H
