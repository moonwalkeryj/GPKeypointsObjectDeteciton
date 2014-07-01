#include "vtkwidget.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/common.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/exceptions.h>

#include <pcl/io/openni_grabber.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/geometry/get_boundary.h>
#include <pcl/geometry/mesh_conversion.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <QtCore>

#include <opencv2/opencv.hpp>

VTKWidget::VTKWidget (QWidget *parent)
    :QVTKWidget(parent),
      mutex_                 (),
      mutex_vis_             (),
      grabber_               (),
      starting_grabber_      (false),
      new_data_connection_   (),
      destructor_called_     (false),
      cloud(new CloudT),
      _isShowPTFilteredCloud  (false),
      _isShowObjectDistance   (false),
      _planeModelSelectDistance(0.03),
      new_cloud_available     (false),
      pviz(new pcl::visualization::PCLVisualizer("", false)),
      _processor(new CloudProcessor<PointT>),
      _passThrough(new pcl::PassThrough<PointT>),
      timer(new QTimer(this)),
      _mainWindow(NULL)

{
    memset(&cam_PosParas[0], 0, 9*sizeof(double));
    cam_PosParas[POSZ] = -3.0; cam_PosParas[VIEWZ] = -1.0; cam_PosParas[UPY] = -1.0;
    _passThroughParas[XMIN] = _passThroughParas[YMIN] = _passThroughParas[ZMIN] = -5.0;
    _passThroughParas[XMAX] = _passThroughParas[YMAX] = _passThroughParas[XMAX] = 5.0;

    // Pass through parameters set defaults
    _passThrough->setFilterFieldName ("x");
    _passThrough->setFilterLimits (0.5, 5.0);
    _passThrough->setFilterFieldName ("y");
    _passThrough->setFilterLimits (0.5, 5.0);
    _passThrough->setFilterFieldName ("z");
    _passThrough->setFilterLimits (0.5, 5.0);

    // Initialize the default plane Model
    _planeModelCoeff.resize(4);
    _planeModelCoeff[0] = -0.0244234;
    _planeModelCoeff[1] = -0.998427;
    _planeModelCoeff[2] = -0.0504705;
    _planeModelCoeff[3] = 0.764362;

    CloudTPtr cloud_xyz (new CloudT);
    {
        for (float y = -1.0f; y <= 1.0f; y += 0.05f)
        {
            for (float z = -1.0f; z <= 1.0f; z += 0.05f)
            {
                PointT point;
                point.x = 2.0f - y;
                point.y = y;
                point.z = z;
                cloud_xyz->points.push_back (point);
            }
        }
        cloud_xyz->width = cloud_xyz->points.size ();
        cloud_xyz->height = 1;
    }

    // this creates and displays a window named "test_viz"
    // upon calling PCLVisualizerInteractor interactor_->Initialize ();
    // how to disable that?
    //        pcl::visualization::PCLVisualizer pviz ("test_viz");
    //        pcl::visualization::PCLVisualizer pviz2("ok");

    pviz->addPointCloud<PointT>(cloud_xyz);
    pviz->setBackgroundColor(0.1, 0.1, 0);
    pviz->addCoordinateSystem(1.0f, 0.0, 0.0, 0.0);
    pviz->initCameraParameters();

    // Initialize window
    this->SetRenderWindow(pviz->getRenderWindow());
    pviz->setupInteractor(this->GetInteractor(), this->GetRenderWindow());
    pviz->getInteractorStyle ()->setKeyboardModifier (pcl::visualization::INTERACTOR_KB_MOD_SHIFT);
    this->update();

    QObject::connect(timer, SIGNAL(timeout()), this, SLOT(updateCloud()));
    QObject::connect(this, SIGNAL(openCloudAvailable()), this, SLOT(updateOpenCloud()));
}

VTKWidget::~VTKWidget()
{
    destructor_called_ = true;
    boost::mutex::scoped_lock lock (mutex_);
    if(timer->isActive()) timer->stop();
    if (grabber_ && grabber_->isRunning ()) grabber_->stop ();
    if (new_data_connection_.connected ())  new_data_connection_.disconnect ();
    this->SetRenderWindow(NULL);
    pviz->removeAllPointClouds();
    delete pviz;
}

void
VTKWidget::startGrabber ()
{
    QtConcurrent::run (boost::bind (&VTKWidget::startGrabberImpl, this));
}

void
VTKWidget::resetCamera()
{
    //pviz->setCameraPosition(cam_PosParas[0], cam_PosParas[1], cam_PosParas[2], cam_PosParas[3], cam_PosParas[4], cam_PosParas[5],cam_PosParas[6], cam_PosParas[7], cam_PosParas[8]);
    pviz->setCameraPosition(cam_PosParas[POSX],cam_PosParas[POSY],cam_PosParas[POSZ],cam_PosParas[UPX],cam_PosParas[UPY],cam_PosParas[UPZ],0);
    //pviz->resetCameraViewpoint();
    update();
}

void
VTKWidget::saveCurrentPointCloud()
{
    mutex_.lock();
    if(cloud->empty()) return;
    //    pcl::PointCloud<pcl::PointXYZRGBA> cloud_out;
    //    std::vector<int> indices;
    //    pcl::removeNaNFromPointCloud(*cloud, cloud_out, indices);
    pcl::io::savePCDFile("current1.pcd", *cloud);
    //    cerr << "---saveCurrentPointCloud: " << indices.size() << " NaN Points removed!"  << endl;
    mutex_.unlock();
}

bool
VTKWidget::loadPCD(std::string str, CloudT& cloudXYZRGBA)
{
    if (pcl::io::loadPCDFile<PointT> (str, cloudXYZRGBA) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return false;
    }
    std::vector<int> indices;
    //pcl::removeNaNFromPointCloud(cloudXYZ, cloudXYZ, indices);
    emit openCloudAvailable();
    cerr << "---loadPCD---Successfully load PCD file......" << endl;
    cerr << "---loadPCD---PCDfile width: " << cloudXYZRGBA.width << "; height: " << cloudXYZRGBA.height << "; num: " << cloudXYZRGBA.size() << "." << endl;
    cerr << "---loadPCD--- " << indices.size() << " NaN indices removed.." << endl;
    return true;
}

bool
VTKWidget::openPCDFile(std::string str)
{
    cerr << "reading file: " << str << endl;
    QtConcurrent::run(boost::bind (&VTKWidget::loadPCD, this, str, *cloud));
    // File type
    return true;
}

bool
VTKWidget::startKinectGrabber()
{
    if (grabber_ && grabber_->isRunning () && new_data_connection_.connected ()) return false;
    QTimer::singleShot (0, this, SLOT (startGrabber ()));
    timer->start(50);
    return true;
}

bool
VTKWidget::closeKinectGrabber()
{
    if (grabber_ && grabber_->isRunning ()) grabber_->stop (); else return false;
    if (new_data_connection_.connected ())  new_data_connection_.disconnect (); else return false;
    if (new_image_connection_.connected()) new_image_connection_.disconnect(); else return false;
    timer->stop();
    return true;
}

void
VTKWidget::setCameraParameters(int pos, double value)
{
    cam_PosParas[pos] = value;
    resetCamera();
}

void
VTKWidget::updateOpenCloud()
{
    CloudTPtr cloud_xyz (new CloudT);
    {
        for (float y = -0.5f; y <= 0.5f; y += 0.01f)
        {
            for (float z = -0.5f; z <= 0.5f; z += 0.01f)
            {
                PointT point;
                point.x = 2.0f - y;
                point.y = y;
                point.z = z;
                cloud_xyz->points.push_back (point);
            }
        }
        cloud_xyz->width = cloud_xyz->points.size ();
        cloud_xyz->height = 1;
    }
    mutex_.lock();
    pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(cloud_xyz, 0, 255, 255);
    pviz->removeAllPointClouds();
    pviz->addPointCloud(cloud_xyz, handler);
    mutex_.unlock();
    cerr << "successfully updated opened PCD file..." << endl;
    update();

}

void VTKWidget::adjustPassThroughValueXMax(int value)
{
    _passThroughParas[XMAX] = ((float)value)/5;
    _passThrough->setFilterFieldName ("x");
    _passThrough->setFilterLimits (_passThroughParas[XMIN], _passThroughParas[XMAX]);
}

void VTKWidget::adjustPassThroughValueYMax(int value)
{
    _passThroughParas[YMAX] = ((float)value)/5;
    _passThrough->setFilterFieldName ("y");
    _passThrough->setFilterLimits (_passThroughParas[YMIN], _passThroughParas[YMAX]);
}

void VTKWidget::adjustPassThroughValueZMax(int value)
{
    _passThroughParas[ZMAX] = ((float)value)/5;
    _passThrough->setFilterFieldName ("z");
    _passThrough->setFilterLimits (_passThroughParas[ZMIN], _passThroughParas[ZMAX]);
}

void VTKWidget::adjustPassThroughValueXMin(int value)
{
    _passThroughParas[XMIN] = ((float)value - 50.0)/5;
    _passThrough->setFilterFieldName ("x");
    _passThrough->setFilterLimits (_passThroughParas[XMIN], _passThroughParas[XMAX]);
}

void VTKWidget::adjustPassThroughValueYMin(int value)
{
    _passThroughParas[YMIN] = ((float)value - 50.0)/5;
    _passThrough->setFilterFieldName ("y");
    _passThrough->setFilterLimits (_passThroughParas[YMIN], _passThroughParas[YMAX]);
}

void VTKWidget::adjustPassThroughValueZMin(int value)
{
    _passThroughParas[ZMIN] = ((float)value - 50.0)/5;
    _passThrough->setFilterFieldName ("z");
    _passThrough->setFilterLimits (_passThroughParas[ZMIN], _passThroughParas[ZMAX]);
}

void
VTKWidget::updateCloud()
{
    static bool count = true;
    if(new_cloud_available&&mutex_.try_lock()){
        if(mutex_vis_.try_lock()){
            // FPS
            FPS_CALC("Cloud Update");

            new_cloud_available = false;

            // display colorful cloud

            CloudTConstPtr temp;
            temp.swap(cloud);
            mutex_.unlock();

            pcl::IndicesPtr inliers_temp(new std::vector<int>);
            CloudTPtr temp_cloud(new CloudT);
            pcl::removeNaNFromPointCloud<PointT>(*temp, *temp_cloud, *inliers_temp);
            //******************
            if (temp_cloud->size() < 3) {mutex_vis_.unlock(); std::cerr << "TOO FAR OR TOO　NEAR!"; return;}

            CloudTPtr filtered_cloud(new CloudT);
            _passThrough->setInputCloud(temp_cloud);
            _passThrough->filter(*filtered_cloud);
            CloudTPtr ground_cloud(new CloudT), no_ground_cloud(new CloudT);

            // Thread : updating ground coefficient
            if(!_processor->isGroundCoeffUpdating() && filtered_cloud->size())
            {
                _processor->setInputCloud(filtered_cloud);
                QtConcurrent::run (boost::bind (&CloudProcessor<PointT>::groundCoeffUpdateThread, _processor));
            }

            // Whether to show passthrough filtered cloud
            if(_isShowPTFilteredCloud)
            {
                temp_cloud = filtered_cloud;
            }

            // Plane Model Coeffs Update
            if(_processor->_mutexGndCef.try_lock())
            {
                _planeModelCoeff = _processor->getGroundCoeffs();
                _processor->_mutexGndCef.unlock();
            }

            // Plane Model Segmentation
            pcl::IndicesPtr inliers(new std::vector<int>);
            boost::shared_ptr<pcl::SampleConsensusModelPlane<PointT>> ground_model(new pcl::SampleConsensusModelPlane<PointT>(temp_cloud));
            ground_model->selectWithinDistance(_planeModelCoeff, _planeModelSelectDistance, *inliers);

            // Plane extraction
            _processor->_IndicesExtractor->setInputCloud (temp_cloud);
            _processor->_IndicesExtractor->setIndices (inliers);
            _processor->_IndicesExtractor->setNegative(true);
            _processor->_IndicesExtractor->filter (*no_ground_cloud);
            _processor->_IndicesExtractor->setNegative(false);
            _processor->_IndicesExtractor->filter(*ground_cloud);

            // Change color of the ground cloud
            /*cerr << "Cloud is organized: " << temp_cloud->isOrganized() << endl;
            cerr << "Ground Cloud is organized: " << ground_cloud->isOrganized() << endl;
            for (int i = 0; i < ground_cloud->height; i++) {
                for (int j = 0; j < ground_cloud->width; j++){
                    (*ground_cloud)(j, i).r = 255;
                }
            }*/
            pcl::visualization::PointCloudColorHandlerCustom<PointT> handler(ground_cloud, 0, 255, 255);

            if (_processor->_mutex_boundingboxes.try_lock()){
                // Draw Clustered Object Bounding Boxes
                pviz->removeAllShapes();
                // bounding box lock
                int iBoxNum = 1;

                for (std::vector<pcl::ModelCoefficients>::const_iterator it = _processor->_BoundingBoxModels.begin();
                     it != _processor->_BoundingBoxModels.end(); it++, iBoxNum++ )
                {
                    std::stringstream bbox_name;
                    bbox_name << iBoxNum;
                    pviz->addCube (*it, bbox_name.str());
                    pviz->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.5, 0.0, bbox_name.str());
                    pviz->setShapeRenderingProperties (pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, bbox_name.str());

                    if (_isShowObjectDistance)
                    {
                        bbox_name << "text";
                        pviz->removeText3D(bbox_name.str());
                        pcl::ModelCoefficients coeff = *it;
                        PointT pos;
                        pos.x = coeff.values.at(0);
                        pos.y = coeff.values.at(1);
                        pos.z = coeff.values.at(2) - coeff.values.at(9)/2;
                        std::stringstream distance;
                        distance.precision(2);
                        distance << coeff.values.at(2);
                        pviz->addText3D<PointT>(distance.str(), pos, 0.1, 1.0, 1.0, 1.0, bbox_name.str());
                    }
                }

                // bounding box unlock
                _processor->_mutex_boundingboxes.unlock();
            }


            // update view
            if (count) {
                pviz->removeAllPointClouds();
                pviz->addPointCloud(no_ground_cloud, "noground");
                pviz->addPointCloud(ground_cloud, handler, "ground");
                count = false;
            }
            pviz->updatePointCloud(no_ground_cloud, "noground");
            pviz->updatePointCloud(ground_cloud, handler, "ground");

            mutex_vis_.unlock();
            update();
            return;
        }
        else {
            mutex_.unlock();
            return;
        }
    }
}

void
VTKWidget::newDataCallbackXYZRGBA (const CloudTConstPtr& cloud_in)
{
    FPS_CALC("New Data Callback");
    if(destructor_called_) return;
    mutex_.lock();
    cloud = cloud_in;

    //std::cerr << "width: " << cloud_in->width << "height: " << cloud_in->height << std::endl;
    if (cloud_in && cloud_in->size() > 0) {
        cv::Mat ones = cv::Mat::zeros(cloud_in->height, cloud_in->width, CV_8UC3);
        //std::cerr << "col: " << ones.cols << "row： " << ones.rows << std::endl;
        for (int row=0;row<ones.rows;row++) {
            unsigned char *data = ones.ptr(row);
            for (int col=0;col<ones.cols;col++) {
                // then use *data for the pixel value, assuming you know the order, RGB etc
                // Note 'rgb' is actually stored B,G,R
                //PointT point = (*cloud_in)(col, row);

                if ( pcl_isfinite ((*cloud_in)(col, row).z)) {

                    //unsigned char data = (*cloud_in)(col, row).z * 50 * 3;
                    *data++ = 0.5 * (*cloud_in)(col, row).b ;
                    *data++ = 0.5 * (*cloud_in)(col, row).g ;
                    *data++ = 0.5 * (*cloud_in)(col, row).r + (*cloud_in)(col, row).z * 25;
                } else {
                    *data++ = 0.5 * (*cloud_in)(col, row).b ;
                    *data++ = 0.5 * (*cloud_in)(col, row).g ;
                    *data++ = 0.5 * (*cloud_in)(col, row).r ;
                }
            }
        }

        cv::imshow("cloudimage", ones);
        cv::waitKey(10);
        new_cloud_available = true;
    }
    mutex_.unlock();


}
bool VTKWidget::isShowPTFilteredCloud() const
{
    return _isShowPTFilteredCloud;
}
float VTKWidget::planeModelSelectDistance() const
{
    return _planeModelSelectDistance;
}

void VTKWidget::setMainWindow(MainWindow *mainWindow)
{
    _mainWindow = mainWindow;
}
bool VTKWidget::isShowObjectDistance() const
{
    return _isShowObjectDistance;
}

void VTKWidget::setIsShowObjectDistance(bool isShowObjectDistance)
{
    _isShowObjectDistance = isShowObjectDistance;
}



void VTKWidget::setPlaneModelSelectDistance(double planeModelSelectDistance)
{
    _planeModelSelectDistance = (float)planeModelSelectDistance;
}


void VTKWidget::setIsShowPTFilteredCloud(bool isShowPTFilteredCloud)
{
    _isShowPTFilteredCloud = isShowPTFilteredCloud;
}


void
VTKWidget::startGrabberImpl ()
{
    boost::mutex::scoped_lock lock (mutex_);
    starting_grabber_ = true;
    lock.unlock ();

    try
    {
        grabber_ = GrabberPtr (new Grabber ());
    }
    catch (const pcl::PCLException& e)
    {
        std::cerr << "ERROR in in_hand_scanner.cpp: " << e.what () << std::endl;
        exit (EXIT_FAILURE);
    }

    lock.lock ();
    if (destructor_called_) return;
    boost::function <void (const CloudTConstPtr&)> new_data_cb = boost::bind (&VTKWidget::newDataCallbackXYZRGBA, this, _1);
    new_data_connection_ = grabber_->registerCallback (new_data_cb);
    //boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&) > new_image_cb = boost::bind (&MainWindow::image_callback, this->_mainWindow, _1);
    //new_image_connection_ = grabber_->registerCallback (new_image_cb);
    grabber_->start ();
    starting_grabber_ = false;
}

void
VTKWidget::mousePressEvent (QMouseEvent *event){
    cerr << "---VTKWidget::mousePressEvent-----" << endl;
    mutex_vis_.lock();
    QVTKWidget::mousePressEvent(event);
}
void
VTKWidget::mouseMoveEvent (QMouseEvent *event){
    QVTKWidget::mouseMoveEvent(event);
}
void
VTKWidget::mouseReleaseEvent (QMouseEvent *event){
    cerr << "---VTKWidget::mouseReleaseEvent-----" << endl;
    mutex_vis_.unlock();
    QVTKWidget::mouseReleaseEvent(event);
}
void
VTKWidget::wheelEvent (QWheelEvent *event){
    QVTKWidget::wheelEvent(event);
}

void VTKWidget::paintEvent(QPaintEvent *event)
{
    //boost::mutex::scoped_lock lock (mutex_vis_);
    QVTKWidget::paintEvent(event);
}

void VTKWidget::resizeEvent(QResizeEvent *event)
{
    QVTKWidget::resizeEvent(event);
}

pcl::visualization::PCLVisualizer *VTKWidget::getPCLVisualizer()
{
    return pviz;
}
