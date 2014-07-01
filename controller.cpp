#include "controller.h"
#include <pcl/range_image/range_image.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/board.h>

#include <pcl/features/shot_omp.h>

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

#include "harris_3d.h"
#include <PCL/keypoints/sift_keypoint.h>
#include <QtConcurrentRun>

void Controller::resetCameraView()
{
    _pcWidget->resetCamera();
}

void Controller::showModel(std::string filename)
{
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool file_is_pcd = true;

    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)	{
        file_is_pcd = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    // Loading file | Works with PCD and PLY files
    pcl::PCLPointCloud2 cloud2;
    if (file_is_pcd) {

        pcl::PCDReader readerPCD;
        readerPCD.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_model);

        std::cout << "Loaded "
                  << _model->width * _model->height
                  << " model from "
                  << filename
                  << std::endl;

        _new_data_available = true;

        _pcWidget->getPCLVisualizer()->removePointCloud("model");
        _pcWidget->getPCLVisualizer()->addPointCloud( _model,"model");

    } else {

        pcl::PLYReader readerPLY;
        readerPLY.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_model);

        std::cout << "Loaded "
                  << _model->width * _model->height
                  << " model from "
                  << filename
                  << std::endl;

        _pcWidget->getPCLVisualizer()->removePointCloud("model");
        _pcWidget->getPCLVisualizer()->addPointCloud( _model,"model");
    }
}

void Controller::showScene(std::string filename)
{
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool file_is_pcd = true;

    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)
    {
        file_is_pcd = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    // Loading file | Works with PCD and PLY files
    pcl::PCLPointCloud2 cloud2;
    if (file_is_pcd) {

        pcl::PCDReader readerPCD;
        readerPCD.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_scene);

        std::cout << "Loaded "
                  << _scene->width * _scene->height
                  << " scene from "
                  << filename
                  << std::endl;

        _new_data_available = true;

        _pcWidget->getPCLVisualizer()->removePointCloud("scene");
        _pcWidget->getPCLVisualizer()->addPointCloud( _scene,"scene");

    } else {

        pcl::PLYReader readerPLY;
        readerPLY.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_scene);

        std::cout << "Loaded "
                  << _scene->width * _scene->height
                  << " scene from "
                  << filename
                  << std::endl;

        _pcWidget->getPCLVisualizer()->removePointCloud("scene");
        _pcWidget->getPCLVisualizer()->addPointCloud( _scene,"scene");
    }
}

void Controller::initModelAndSceneResolution()
{
    //
    //  Set up resolution invariance
    //
    if (_use_cloud_resolution_)
    {
        float resolution = static_cast<float> (computeCloudResolution (_model));
        if (resolution != 0.0f)
        {
            _model_ss_   *= resolution;
            _scene_ss_   *= resolution;
            _rf_rad_     *= resolution;
            _descr_rad_  *= resolution;
            _cg_size_    *= resolution;
        }

        std::cout << "Model resolution:       " << resolution << std::endl;
        std::cout << "Model sampling size:    " << _model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << _scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << _rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << _descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << _cg_size_ << std::endl << std::endl;
    }
}

void Controller::detectModelGPKeypoints(QString strGeno)
{
    //
    // Using GP Keypoints Detector
    //
    float model_resol = computeCloudResolution(_model);
    TrainingData::extractFeatrues(_model, _features, 10.*model_resol);

    // populize domdocument
    QDomDocument doc;
    if (!doc.setContent(strGeno))
    {
        qDebug() << "Fail to populize domdocument...";
        return;
    }

    cv::Mat_<float> result;
    cv::Mat norm;

    // get feature point detection result
    result = preOrder( doc.firstChildElement().firstChildElement() );

    // normalizing to period [0, 255]
    cv::normalize( result, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );
    std::cout << "****++++****" << std::endl;
    pcl::PointCloud<int> indices;
    int cnt = 0;
    for (int i=0; i < norm.cols; i++)
    {
        if (((uchar*)norm.data)[i] > _gpThreshold)
        {
            cnt++;
            indices.push_back(i);
        }
    }
    std::cout << "****++++****" << std::endl;
    std::cout << cnt << " keypoints detected......." << std::endl;

    pcl::copyPointCloud (*_model, indices.points, *_model_keypoints);

    _pcWidget->getPCLVisualizer()->removePointCloud("model_keypoints");
    _pcWidget->getPCLVisualizer()->addPointCloud(_model_keypoints, "model_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "model_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0.0, "model_keypoints");
    _pcWidget->viewupdate();
}

void Controller::detectSceneGPKeypoints(QString strGeno)
{
    //
    // Using GP Keypoints Detector
    //
    float scene_resol = computeCloudResolution(_scene);
    TrainingData::extractFeatrues(_scene, _features, scene_resol);

    // populize domdocument
    QDomDocument doc;
    if (!doc.setContent(strGeno))
    {
        qDebug() << "Fail to populize domdocument...";
        return;
    }

    cv::Mat_<float> result;
    cv::Mat norm;

    // get feature point detection result
    result = preOrder( doc.firstChildElement().firstChildElement() );

    // normalizing to period [0, 255]
    cv::normalize( result, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );

    pcl::PointCloud<int> indices;
    int cnt = 0;
    for (int i=0; i < norm.cols; i++)
    {
        if (((uchar*)norm.data)[i] > _gpThreshold)
        {
            cnt++;
            indices.push_back(i);
        }
    }

    std::cout << cnt << " keypoints detected......." << std::endl;

    pcl::copyPointCloud (*_scene, indices.points, *_scene_keypoints);

    _pcWidget->getPCLVisualizer()->removePointCloud("scene_keypoints");
    _pcWidget->getPCLVisualizer()->addPointCloud(_scene_keypoints, "scene_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "scene_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0.0, "scene_keypoints");
    _pcWidget->viewupdate();

}

void Controller::computeModelDescriptors()
{

    //
    //  Compute SHOT Descriptor for keypoints
    //

    //  Compute Normals
    //
    pcl::NormalEstimation<PointT, NormalT> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (_model);
    norm_est.compute (*_model_normals);


    //    pcl::SHOTEstimationOMP<PointT, NormalT, DescriptorType> descr_est;
    //    descr_est.setRadiusSearch (_descr_rad_);

    //    descr_est.setInputCloud (_model_keypoints);
    //    descr_est.setInputNormals (_model_normals);
    //    descr_est.setSearchSurface (_model);
    //    descr_est.compute (*_model_descriptors);
}

void Controller::computeSceneDescriptors()
{
    //
    //  Compute SHOT Descriptor for keypoints
    //

    //  Compute Normals
    //
    pcl::NormalEstimation<PointT, NormalT> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (_scene);
    norm_est.compute (*_scene_normals);


    //    pcl::SHOTEstimationOMP<PointT, NormalT, DescriptorType> descr_est;
    //    descr_est.setRadiusSearch (_descr_rad_);

    //    descr_est.setInputCloud (_scene_keypoints);
    //    descr_est.setInputNormals (_scene_normals);
    //    descr_est.setSearchSurface (_scene);
    //    descr_est.compute (*_scene_descriptors);
}




void Controller::appendTrainingFeatures()
{
    if(_cloudPtr->points.empty())
    {
        std::cout << "No cloud to add" << std::endl;
    }
    _trainingData.appendFeaturesAndTarget(_cloudPtr);
}

void Controller::toggleShowHarrisKeypoints(bool ischecked)
{
    if(ischecked)
    {
        detectHarris3DKeypoints();
    }
    else
    {
        removeHarris3DKeypoints();
    }
}

void Controller::toggleShowISSKeypoints(bool ischecked)
{
    if(ischecked)
    {
        detectISSKeypoints();
    }
    else
    {
        removeISSKeypoints();
    }
}

void Controller::toggleShowGPKeypoints(bool ischecked, QString strGeno)
{
    if(ischecked)
    {
        detectGPKeypoints(strGeno);
    }
    else
    {
        removeGPKeypoints();
    }
}

Controller::Controller( QPointCloudWidget * pointcloudWidget):
    _trainingData(this),
    _mutex(),
    _cloudPtr(new CloudT),
    _cloudKeypoints(new CloudT),
    _new_data_available(false),
    _pcWidget(pointcloudWidget),
    _model (new pcl::PointCloud<PointT> ()),
    _model_keypoints (new pcl::PointCloud<PointT> ()),
    _scene (new pcl::PointCloud<PointT> ()),
    _scene_keypoints (new pcl::PointCloud<PointT> ()),
    _model_normals (new pcl::PointCloud<NormalT> ()),
    _scene_normals (new pcl::PointCloud<NormalT> ()),
    _model_descriptors (new pcl::PointCloud<DescriptorType> ()),
    _scene_descriptors (new pcl::PointCloud<DescriptorType> ()),
    //    _model_ss_ (0.01f),
    //    _scene_ss_ (0.03f),
    //    _rf_rad_ (0.015f),
    //    _descr_rad_ (0.02f),
    //    _cg_size_ (0.01f),
    //    _cg_thresh_ (5.0f),
    _model_ss_ (10.f),
    _scene_ss_ (20.f),
    _rf_rad_ (15.f),
    _descr_rad_ (20.f),
    _cg_size_ (10.f),
    _cg_thresh_ (4.0f),
    _show_keypoints_ (false),
    _show_correspondences_ (false),
    _use_cloud_resolution_ (false),
    _use_hough_ (true)

{
    cv::Mat temp;
    //populate desired corner mask
    cv::FileStorage storagemsk("corner_mask.xml", cv::FileStorage::READ); storagemsk["f"] >> temp; storagemsk.release();
    //_desired_cormsk = cv::imread("corner_mask.yml");
    cv::Mat_<uchar> _matTargetPts;
    _matTargetPts = temp.clone();

    std::cout << "Target: rows: " << _matTargetPts.rows << "cols: " << _matTargetPts.cols << std::endl;

}

void Controller::showPointCloud(std::string filename)
{
    //    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    //    if (pcl::io::loadPCDFile<PointT> (filename, *cloud) == -1) //* load the file
    //    {
    //        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    //        return;
    //    }

    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool file_is_pcd = true;

    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)	{
        file_is_pcd = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    // Loading file | Works with PCD and PLY files
    pcl::PCLPointCloud2 cloud2;
    if (file_is_pcd) {
        pcl::PCDReader readerPCD;
        readerPCD.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_cloudPtr);

        std::cout << "Loaded "
                  << _cloudPtr->width * _cloudPtr->height
                  << " data points from "
                  << filename
                  << std::endl;

        _new_data_available = true;
        _pcWidget->showPointCloud(_cloudPtr);

    } else {
        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PLYReader readerPLY;
        readerPLY.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_cloudPtr);

        std::cout << "Loaded "
                  << _cloudPtr->width * _cloudPtr->height
                  << " data points from "
                  << filename
                  << std::endl;

        _pcWidget->getPCLVisualizer()->removePointCloud("cloud");
        _pcWidget->getPCLVisualizer()->addPointCloud( _cloudPtr,pcl::visualization::PointCloudColorHandlerCustom<PointT> (_cloudPtr, 255.0, 255.0, 255.0), "cloud");
        //_pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "keypoint");
        _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 1.0, 1.0, "cloud");
        return;
    }

}

void Controller::showKeypoints(std::string filename)
{
    int result = filename.find("pcd");
    bool isPCDFile = true;
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)	{
        isPCDFile = false;
        std::cout << "this is xml file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    if (isPCDFile)
    // read keypoints from pcd
    {
        pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

        if (pcl::io::loadPCDFile<PointT> (filename, *cloud) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            return;
        }
        std::cout << "Loaded "
                  << cloud->width * cloud->height
                  << " data points from "
                  << filename
                  << std::endl;
        pcl::PCDReader reader;
        pcl::PCLPointCloud2 cloud2;
        reader.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_cloudKeypoints);
    }
    // read keypoints from xml file
    else
    {
        cv::Mat temp;
        // populate desired corner mask
        // cv::FileStorage storagemsk(filename, cv::FileStorage::READ);
        cv::FileStorage storagemsk(filename, cv::FileStorage::READ);
        if(storagemsk.isOpened()) std::cout << "File is opened!" << std::endl;
        else std::cout << "Not opened" << std::endl;

        storagemsk["f"] >> temp;
        storagemsk.release();

        std::cout << "Target: rows: " << temp.rows << "cols: " << temp.cols << std::endl;
        cv::Mat_<uchar> _matTargetPts;
        _matTargetPts = temp.clone();

        std::cout << "Target: rows: " << _matTargetPts.rows << "cols: " << _matTargetPts.cols << std::endl;

        // find index of keypoints
        pcl::PointCloud<int> indices;
        int cnt = 0;
        for (int i=0; i < _matTargetPts.cols; i++)
        {
            if (((uchar*)_matTargetPts.data)[i] == 255)
            {
                indices.push_back(i);
                cnt++;
            }
        }
        std::cout << cnt << " keypoints read......." << std::endl;
        pcl::copyPointCloud (*_cloudPtr, indices.points, *_cloudKeypoints);
    }

    // Display KeyPoints
    _pcWidget->getPCLVisualizer()->removePointCloud("keypoint");

    _pcWidget->getPCLVisualizer()->addPointCloud( _cloudKeypoints, "keypoint");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "keypoint");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1.0, "keypoint");

    _pcWidget->viewupdate();
}

void Controller::removeKeypoints()
{
    _pcWidget->getPCLVisualizer()->removePointCloud("keypoint");
    _pcWidget->viewupdate();
}

double Controller::computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud)
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);
    pcl::search::KdTree<PointT> tree;
    tree.setInputCloud (cloud);

    for (size_t i = 0; i < cloud->size (); ++i)
    {
        if (! pcl_isfinite ((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
        if (nres == 2)
        {
            res += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }

    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

void Controller::setGpThreshold(int gpThreshold)
{
    _gpThreshold = gpThreshold;
}

void Controller::savePointCloud(QString _filename)
{
    if (_cloudPtr->points.empty()) { std::cout << "No cloud to save!!!" << std::endl; return; }

    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool isPCDFile = true;
    std::string filename = _filename.toStdString();
    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)	{
        isPCDFile = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    if (isPCDFile)
    {
        pcl::PCDWriter pcdWriter;
        pcdWriter.write(filename, *_cloudPtr);
    }
    else
    {
        pcl::PLYWriter plyWriter;
        plyWriter.write(filename, *_cloudPtr);
    }
}

void Controller::saveKeypoints(QString filename)
{
    if (_cloudKeypoints->points.empty()) { std::cout << "No keypoints to save!!!" << std::endl; return; }
    pcl::PCDWriter cloudWriter;
    cloudWriter.write(filename.toAscii().data(), *_cloudKeypoints);
}

void Controller::detectHarris3DKeypoints()
{
    //**********Harris*********
    // Harris Initialization
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZI> ());

    // {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod
    //    std::string save_file[5] = {"harris_keypoints.pcd",
    //                                "noble_keypoints.pcd",
    //                                "lowe_keypoints.pcd",
    //                                "tomasi_keypoints.pcd",
    //                                "curvature_keypoints.pcd"};
    //    for (int i_method = 0; i_method < 1; i_method++)
    //    {
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

    // Using Harris3D to Compute Keypoints
    pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI> harris_detector;
    harris_detector.setNonMaxSupression(true);
    //harris_detector.setMethod((pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::ResponseMethod)i_method);
    harris_detector.setRefine(false);
    harris_detector.setSearchMethod(tree);
    harris_detector.setRadius(0.1f);
    harris_detector.setInputCloud(_cloudPtr);
    harris_detector.compute(*keypoints);

    //std::cout << "Keyp points size:" << harris_detector.keypoints_indices_->indices.size();
    std::cerr <<  keypoints->points.size () << " keypoints detected" << std::endl;

    // Iterate through the keypoints cloud
    pcl::PointXYZ tmp;
    double max = 0, min = 0;
    for(pcl::PointCloud<pcl::PointXYZI>::iterator i = keypoints->begin(); i!= keypoints->end(); i++){
        tmp = pcl::PointXYZ((*i).x,(*i).y,(*i).z);
        if ((*i).intensity>max ){
            //std::cout << (*i) << " coords: " << (*i).x << ";" << (*i).y << ";" << (*i).z << std::endl;
            max = (*i).intensity;
        }
        if ((*i).intensity<min){
            min = (*i).intensity;
        }
        //keypoints3D->push_back(tmp);
    }
    std::cout << "maximal responce: "<< max << " min responce:  "<< min << std::endl;

    // Harris
    // pcl::io::savePCDFileASCII (save_file[i_method-1], *keypoints);
    // std::cerr << "Saved " << keypoints->points.size () << " data points to" << save_file[i_method-1] << std::endl;
    // }
    //
    /*
    harris_detector.setMethod(pcl::HarrisKeypoint3D::LOWE);
    harris_keypoints->clear();
    harris_detector.setInputCloud(_cloudPtr);
    harris_detector.compute(*harris_keypoints);
    harris_detector.setInputCloud(_cloudPtr);
    harris_detector.compute(*harris_keypoints);
    */
    //

    _pcWidget->getPCLVisualizer()->removePointCloud("harris3d_keypoints");
    _pcWidget->getPCLVisualizer()->addPointCloud(keypoints, ColorHandlerT3 (keypoints, 0.0, 255.0, 255.0), "harris3d_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "harris3d_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1.0, 1.0, "harris3d_keypoints");

    _pcWidget->viewupdate();
}

void Controller::removeHarris3DKeypoints()
{
    _pcWidget->getPCLVisualizer()->removePointCloud("harris3d_keypoints");
    _pcWidget->viewupdate();
}

void Controller::detectISSKeypoints()
{
    //**********ISS*********
    // ISS Initialization
    pcl::PointCloud<PointT>::Ptr iss_keypoints (new pcl::PointCloud<PointT> ());
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

    // Compute model_resolution
    double model_resolution;
    model_resolution = computeCloudResolution(_cloudPtr);

    // Using ISS to Compute Keypoints
    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    iss_detector.setSearchMethod (tree);
    iss_detector.setSalientRadius (6 * model_resolution);
    iss_detector.setNonMaxRadius (4 * model_resolution);
    iss_detector.setThreshold21 (0.975);
    iss_detector.setThreshold32 (0.975);
    iss_detector.setMinNeighbors (5);
    iss_detector.setNumberOfThreads (4);
    iss_detector.setInputCloud (_cloudPtr);
    iss_detector.compute (*iss_keypoints);

    _pcWidget->getPCLVisualizer()->removePointCloud("iss_keypoints");
    _pcWidget->getPCLVisualizer()->addPointCloud(iss_keypoints, "iss_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "iss_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0., 1.0, 0.0, "iss_keypoints");
    _pcWidget->viewupdate();
}

void Controller::removeISSKeypoints()
{
    _pcWidget->getPCLVisualizer()->removePointCloud("iss_keypoints");
    _pcWidget->viewupdate();
}

void Controller::detectGPKeypoints(QString strGeno)
{

    pcl::GPKeypoint3D<PointT, pcl::PointXYZI> gpDetector;
    // Harris Initialization
    pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

    gpDetector.setGeno(strGeno);
    gpDetector.setNonMaxSupression(true);
    //harris_detector.setMethod((pcl::HarrisKeypoint3D<PointT, pcl::PointXYZI>::ResponseMethod)i_method);
    gpDetector.setRefine(false);
    gpDetector.setSearchMethod(tree);
    float resol = computeCloudResolution(_cloudPtr);
    gpDetector.setThreshold(_gpThreshold);
    gpDetector.setRadius(resol*10.);
    gpDetector.setInputCloud(_cloudPtr);
    gpDetector.compute(*keypoints);

    std::cout << "gp keypoints size:" << keypoints->size() << std::endl;
    /*
    std::cout << cnt << " keypoints detected......." << std::endl;
    pcl::PointCloud<PointT>::Ptr keypoints (new pcl::PointCloud<PointT> ());

    pcl::copyPointCloud (*_cloudPtr, indices.points, *keypoints);
    */
    //
    // detect keypoints for model
    //
    if (!_model->points.empty())
    {
        pcl::GPKeypoint3D<PointT, pcl::PointXYZI> gpDetectorModel;
//        pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints_m (new pcl::PointCloud<pcl::PointXYZI> ());
        // Harris Initialization
        pcl::search::KdTree<PointT>::Ptr tree_m (new pcl::search::KdTree<PointT> ());

        gpDetectorModel.setGeno(strGeno);
        gpDetectorModel.setNonMaxSupression(true);

        gpDetectorModel.setRefine(false);
        gpDetectorModel.setSearchMethod(tree_m);
        float model_resol = computeCloudResolution(_model);
        gpDetectorModel.setRadius(10.*model_resol);
        gpDetectorModel.setThreshold(_gpThreshold);
        gpDetectorModel.setInputCloud(_model);
//        gpDetectorModel.compute(*keypoints_m);
        pcl::PointCloud<int> indices;
        gpDetectorModel.computeKeypointsIndices(indices);

        pcl::copyPointCloud (*_model, indices.points, *_model_keypoints);
        std::cout << "model keypoints size:" << _model_keypoints->size() << std::endl;
    }
    //
    // detect keypoints for scene
    //
    if (!_scene->points.empty())
    {
        pcl::GPKeypoint3D<PointT, pcl::PointXYZI> gpDetectorScene;
        // Harris Initialization
        pcl::search::KdTree<PointT>::Ptr tree_s (new pcl::search::KdTree<PointT> ());

        gpDetectorScene.setGeno(strGeno);
        gpDetectorScene.setNonMaxSupression(true);
        gpDetectorScene.setRefine(false);
        gpDetectorScene.setSearchMethod(tree_s);
        float scene_resol = computeCloudResolution(_scene);
        gpDetectorScene.setRadius(10.*scene_resol);
        gpDetectorScene.setThreshold(_gpThreshold);
        gpDetectorScene.setInputCloud(_scene);

        pcl::PointCloud<int> indices;
        gpDetectorScene.computeKeypointsIndices(indices);

        pcl::copyPointCloud (*_scene, indices.points, *_scene_keypoints);
        std::cout << "Scene keypoints size:" << _scene_keypoints->size() << std::endl;

    }
    //    display

    _pcWidget->getPCLVisualizer()->removePointCloud("geno_keypoints");
    //_pcWidget->getPCLVisualizer()->addPointCloud(keypoints, "geno_keypoints");
    _pcWidget->getPCLVisualizer()->addPointCloud(keypoints, ColorHandlerT3 (keypoints, 0.0, 255.0, 255.0), "geno_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "geno_keypoints");
    _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0.0, "geno_keypoints");
    _pcWidget->viewupdate();

    std::cout << "Model total points: " << _model->size () << "; Selected Keypoints: " << _model_keypoints->size () << std::endl;
    std::cout << "Scene total points: " << _scene->size () << "; Selected Keypoints: " << _scene_keypoints->size () << std::endl;
}

void Controller::removeGPKeypoints()
{
    _pcWidget->getPCLVisualizer()->removePointCloud("geno_keypoints");
    _pcWidget->viewupdate();
}

void Controller::generateArtificialCloud()
{
    cv::Mat_<uchar> _desired_cormsk;

    int width = 0;
    CloudTPtr cloud_xyz (new CloudT);
    CloudTPtr cloud_keypts (new CloudT);
    {
        int counter = 0;
        float global_max_z = 3;
        int count = 0;
        for ( float k = 0.4; k < global_max_z; counter++){

            float max_z = k;
            float offset = global_max_z - k;
            int i = counter % 6;
            int j = counter / 6;
            for (float x = -1.25f; x <= 1.25f; x += 0.05f, width++)
            {
                for (float y = -1.25f; y <= 1.25f; y += 0.05f)
                {
                    PointT point;
                    if ( x >= -1.0f && x <= 1.0f && y >= -1.0f && y <= 1.0f)
                    {
                        // xyz coordinates
                        point.x = x + i * (2.5) + 1.25;
                        point.y = y + j * (2.5) + 1.25;
                        float z_t = k*std::sqrt(x*x + y*y);
                        point.z = (z_t > max_z?max_z:z_t) + offset;

                        // rgb color
                        uint8_t r=0, g=0,b=0;
                        if (x > -0.04f && x < 0.04f && y >= -0.04f && y <= 0.04f)
                        {

                            PointT keypoint;

                            keypoint.x = x + i * (2.5) + 1.25;
                            keypoint.y = y + j * (2.5) + 1.25;
                            float z_t = k*std::sqrt(x*x + y*y);
                            keypoint.z = (z_t > max_z?max_z:z_t) + offset;

                            cloud_keypts->points.push_back(keypoint);
                            g = 0;
                            r = 0;
                            b = 255;
                            ((uchar*)_desired_cormsk.data)[count] = 255;
                            //std::cout << "center point" << std::endl;
                        }
                        else
                        {
                            r = (uint8_t)(point.x/18 * 70 + 170);
                            g = 80;
                            b = 50;
                        }
                        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                        point.rgb = rgb;
                    }
                    else
                    {
                        // xyz coordinates
                        point.x = x + i * (2.5) + 1.25;
                        point.y = y + j * (2.5) + 1.25;
                        point.z = max_z + offset;

                        // rgb color
                        uint8_t r=0, g=0,b=0;
                        r = (uint8_t)(point.x/18 * 70 + 170);
                        g = 80;
                        b = 50;
                        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
                        point.rgb = rgb;
                    }


                    cloud_xyz->points.push_back (point);
                    count++;
                }
            }

            if(k<1.0) k += 0.045;
            else k += 0.05;
        }

        int c = 0;
        for (int i=0; i < cloud_xyz->points.size(); i++)
        {
            if((uchar*)(_desired_cormsk.data)[i]) c++;
        }
        width /= 54;
        std::cout << "corner number mask:" << c << endl;
        std::cout << "width:" << width << "; g_w: " << width*9 << "; g_h: " << width*6 << "; total: " << width*width*54 << endl;

        cv::FileStorage fsmsk("corner_mask.xml", cv::FileStorage::WRITE ); fsmsk << "mask" << _desired_cormsk; fsmsk.release();

        // save cloud
        cloud_xyz->width = cloud_xyz->points.size ();
        cloud_xyz->height = 1;
        pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud_xyz);
        std::cerr << "Saved " << cloud_xyz->points.size () << " data points to test_pcd.pcd." << std::endl;

        // save keypoint cloud
        cloud_keypts->width = cloud_keypts->points.size ();
        cloud_keypts->height = 1;
        pcl::io::savePCDFileASCII ("test_pcd_keypoints.pcd", *cloud_keypts);
        std::cerr << "Saved " << cloud_keypts->points.size () << " key points to test_pcd_keypoints.pcd." << std::endl;

    }
}

void Controller::rotateX()
{
    if (_cloudPtr->points.empty()) return;
    // Defining rotation matrix and translation vector
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity(); // We initialize this matrix to a null transformation.

    //			|-------> This column is the translation
    //	| 1 0 0 x |  \
    //	| 0 1 0 y |   }-> The identity 3x3 matrix (no rotation)
    //	| 0 0 1 z |  /
    //	| 0 0 0 1 |    -> We do not use this line

    // Defining a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    float theta = M_PI/36; // The angle of rotation in radians
    transformation_matrix (0,0) = cos(theta);
    transformation_matrix (0,1) = -sin(theta);
    transformation_matrix (1,0) = sin(theta);
    transformation_matrix (1,1) = cos(theta);

    // Defining a translation of 2 meters on the x axis.
    //transformation_matrix (0,3) = 2.5;

    // Printing the transformation
    printf ("\nThis is the rotation matrix :\n");
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (0,0), transformation_matrix (0,1), transformation_matrix (0,2));
    printf ("R = | %6.1f %6.1f %6.1f | \n", transformation_matrix (1,0), transformation_matrix (1,1), transformation_matrix (1,2));
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (2,0), transformation_matrix (2,1), transformation_matrix (2,2));
    printf ("\nThis is the translation vector :\n");
    printf ("t = < %6.1f, %6.1f, %6.1f >\n", transformation_matrix (0,3), transformation_matrix (1,3), transformation_matrix (2,3));

    // Executing the transformation
    pcl::PointCloud<PointT>::Ptr transformed_cloud (new pcl::PointCloud<PointT> ());	// A pointer on a new cloud
    pcl::transformPointCloud (*_cloudPtr, *transformed_cloud, transformation_matrix);

    _pcWidget->showPointCloud(transformed_cloud);
}

void Controller::rotate(int degrees, int axis)
{
    std::cout << "axis: " << axis << std::endl;
    if (_cloudPtr->points.empty()) return;
    // Defining rotation matrix and translation vector
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity(); // We initialize this matrix to a null transformation.

    //			|-------> This column is the translation
    //	| 1 0 0 x |  \
    //	| 0 1 0 y |   }-> The identity 3x3 matrix (no rotation)
    //	| 0 0 1 z |  /
    //	| 0 0 0 1 |    -> We do not use this line

    // Defining a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
    float theta = ((double)degrees/180)*M_PI; // The angle of rotation in radians

    switch (axis ){
    case 0: // z axis
    {
        transformation_matrix (0,0) = cos(theta);
        transformation_matrix (0,1) = -sin(theta);
        transformation_matrix (1,0) = sin(theta);
        transformation_matrix (1,1) = cos(theta);
        break;
    }
    case 1: // y axis
    {
        transformation_matrix (0,0) = cos(theta);
        transformation_matrix (0,2) = -sin(theta);
        transformation_matrix (2,0) = sin(theta);
        transformation_matrix (2,2) = cos(theta);
        break;
    }
    case 2: // x axis
    {
        transformation_matrix (1,1) = cos(theta);
        transformation_matrix (1,2) = -sin(theta);
        transformation_matrix (2,1) = sin(theta);
        transformation_matrix (2,2) = cos(theta);
        break;
    }
    }

    // Printing the transformation
    printf ("\nThis is the rotation matrix :\n");
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (0,0), transformation_matrix (0,1), transformation_matrix (0,2));
    printf ("R = | %6.1f %6.1f %6.1f | \n", transformation_matrix (1,0), transformation_matrix (1,1), transformation_matrix (1,2));
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (2,0), transformation_matrix (2,1), transformation_matrix (2,2));
    printf ("\nThis is the translation vector :\n");
    printf ("t = < %6.1f, %6.1f, %6.1f >\n", transformation_matrix (0,3), transformation_matrix (1,3), transformation_matrix (2,3));

    // Executing the transformation
    // pcl::PointCloud<PointT>::Ptr transformed_cloud (new pcl::PointCloud<PointT> ());	// A pointer on a new cloud
    pcl::transformPointCloud (*_cloudPtr, *_cloudPtr, transformation_matrix);

    _pcWidget->showPointCloud(_cloudPtr);
}

void Controller::translate(int distance, int axis)
{
    std::cout << "axis: " << axis << std::endl;
    if (_cloudPtr->points.empty()) return;
    // Defining rotation matrix and translation vector
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity(); // We initialize this matrix to a null transformation.

    //			|-------> This column is the translation
    //	| 1 0 0 x |  \
    //	| 0 1 0 y |   }-> The identity 3x3 matrix (no rotation)
    //	| 0 0 1 z |  /
    //	| 0 0 0 1 |    -> We do not use this line

    // Defining a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)

    switch (axis ){
    case 0: // z axis
    {
        transformation_matrix (0,3) = distance;
        break;
    }
    case 1: // y axis
    {
        transformation_matrix (1,3) = distance;
        break;
    }
    case 2: // x axis
    {
        transformation_matrix (2,3) = distance;
        break;
    }
    }
    // Printing the transformation
    printf ("\nThis is the rotation matrix :\n");
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (0,0), transformation_matrix (0,1), transformation_matrix (0,2));
    printf ("R = | %6.1f %6.1f %6.1f | \n", transformation_matrix (1,0), transformation_matrix (1,1), transformation_matrix (1,2));
    printf ("    | %6.1f %6.1f %6.1f | \n", transformation_matrix (2,0), transformation_matrix (2,1), transformation_matrix (2,2));
    printf ("\nThis is the translation vector :\n");
    printf ("t = < %6.1f, %6.1f, %6.1f >\n", transformation_matrix (0,3), transformation_matrix (1,3), transformation_matrix (2,3));

    // Executing the transformation
    pcl::transformPointCloud (*_cloudPtr, *_cloudPtr, transformation_matrix);

    _pcWidget->showPointCloud(_cloudPtr);
}

void Controller::objectDetectionTest()
{
    // Calculate resolution
    float resolution_m = static_cast<float> (computeCloudResolution (_model));
    float resolution_s = static_cast<float> (computeCloudResolution (_scene));
    float resolution = resolution_m < resolution_s ? resolution_m:resolution_s;
    if (true)
    {

        if (resolution != 0.0f)
        {
            _model_ss_   = 10. * resolution_m;
            _scene_ss_   = 10. * resolution_s;
            _rf_rad_     = 15. * resolution;
            _descr_rad_  = 20. * resolution;
            _cg_size_    = 10. * resolution;
        }
        std::cout << "***AFTER MODIF***" << std::endl;
        std::cout << "Model resolution:       " << resolution_m << std::endl;
        std::cout << "Scene resolution:       " << resolution_s << std::endl;
        std::cout << "Resolution:             " << resolution << std::endl;
        std::cout << "Model sampling size:    " << _model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << _scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << _rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << _descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << _cg_size_ << std::endl << std::endl;
    }

    //
    //  Compute Normals
    //
    pcl::NormalEstimation<PointT, NormalT> norm_est;
    norm_est.setKSearch (10);
    norm_est.setInputCloud (_model);
    norm_est.compute (*_model_normals);

    norm_est.setInputCloud (_scene);
    norm_est.compute (*_scene_normals);

    //
    //  Extract keypoints
    //


    switch (2)
    {
    case 1:
    {
        // Using uniform sampling to compute keypoints
        pcl::PointCloud<int> sampled_indices;
        pcl::UniformSampling<PointT> uniform_sampling;

        uniform_sampling.setInputCloud (_model);
        uniform_sampling.setRadiusSearch (_model_ss_);
        uniform_sampling.compute (sampled_indices);
        pcl::copyPointCloud (*_model, sampled_indices.points, *_model_keypoints);

        // Using uniform sampling to compute keypoints
        uniform_sampling.setInputCloud (_scene);
        uniform_sampling.setRadiusSearch (_scene_ss_);
        uniform_sampling.compute (sampled_indices);
        pcl::copyPointCloud (*_scene, sampled_indices.points, *_scene_keypoints);

        break;
    }
    case 2:
    {
        // Using ISS to Compute Keypoints
        pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
        pcl::search::KdTree<PointT>::Ptr tree1 (new pcl::search::KdTree<PointT> ());
        iss_detector.setSearchMethod (tree1);
        iss_detector.setSalientRadius (6 * resolution_m);
        iss_detector.setNonMaxRadius (4 * resolution_m);
        iss_detector.setThreshold21 (0.975);
        iss_detector.setThreshold32 (0.975);
        iss_detector.setMinNeighbors (5);
        iss_detector.setNumberOfThreads (4);
        iss_detector.setInputCloud (_model);
        iss_detector.compute (*_model_keypoints);

        // Using ISS to Compute Keypoints
        pcl::search::KdTree<PointT>::Ptr tree2 (new pcl::search::KdTree<PointT> ());
        iss_detector.setSearchMethod (tree2);
        iss_detector.setSalientRadius (6 * resolution_s);
        iss_detector.setNonMaxRadius (4 * resolution_s);
        iss_detector.setThreshold21 (0.975);
        iss_detector.setThreshold32 (0.975);
        iss_detector.setMinNeighbors (5);
        iss_detector.setNumberOfThreads (4);
        iss_detector.setInputCloud (_scene);
        iss_detector.compute (*_scene_keypoints);

        break;
    }
    case 3:
    {
        // Using GP detector to compute keypoints

        break;
    }

    }

    std::cout << "Model total points: " << _model->size () << "; Selected Keypoints: " << _model_keypoints->size () << std::endl;
    std::cout << "Scene total points: " << _scene->size () << "; Selected Keypoints: " << _scene_keypoints->size () << std::endl;

    //
    //  Compute Descriptor for keypoints
    //
    //***Using SHOT　Descriptor***BEGIN
    pcl::SHOTEstimation<PointT, NormalT, DescriptorType> descr_est;
    descr_est.setRadiusSearch (_descr_rad_);

    descr_est.setInputCloud (_model_keypoints);
    descr_est.setInputNormals (_model_normals);
    descr_est.setSearchSurface (_model);
    descr_est.compute (*_model_descriptors);

    descr_est.setInputCloud (_scene_keypoints);
    descr_est.setInputNormals (_scene_normals);
    descr_est.setSearchSurface (_scene);
    descr_est.compute (*_scene_descriptors);
    //***Using SHOT　Descriptor***END

    //***Using PFH　Descriptor***BEGIN
    //    //pcl::SHOTEstimation<PointT, NormalT, DescriptorType> descr_est;
    //    pcl::PFHEstimation<PointT, NormalT, pcl::PFHSignature125> descr_est;
    //    descr_est.setRadiusSearch (_descr_rad_);
    //    descr_est.setInputCloud (_model_keypoints);
    //    descr_est.setInputNormals (_model_normals);
    //    // Create an empty kdtree representation, and pass it to the PFH estimation object.
    //    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    //    pcl::search::KdTree<PointT>::Ptr tree1 (new pcl::search::KdTree<PointT> ());
    //    //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
    //    descr_est.setSearchMethod (tree1);
    //    descr_est.setSearchSurface (_model);
    //    descr_est.compute (*_model_descriptors);

    //    descr_est.setInputCloud (_scene_keypoints);
    //    descr_est.setInputNormals (_scene_normals);
    //    // Create an empty kdtree representation, and pass it to the PFH estimation object.
    //    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    //    pcl::search::KdTree<PointT>::Ptr tree2 (new pcl::search::KdTree<PointT> ());
    //    //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
    //    descr_est.setSearchMethod (tree2);
    //    descr_est.setSearchSurface (_scene);
    //    descr_est.compute (*_scene_descriptors);
    //***Using PFH　Descriptor***END

    //
    //  Find Model-Scene Correspondences with KdTree
    //
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud (_model_descriptors);

    //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < _scene_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);
        std::vector<float> neigh_sqr_dists (1);

        if (!pcl_isfinite (_scene_descriptors->at (i).descriptor[0])) //skipping NaNs
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (_scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        {
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back (corr);
        }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

    //
    //  Actual Clustering
    //
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //  Using Hough3D
    if (true)//use_hough_)
    {
        //
        //  Compute (Keypoints) Reference Frames only for Hough
        //
        pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
        pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

        pcl::BOARDLocalReferenceFrameEstimation<PointT, NormalT, RFType> rf_est;
        rf_est.setFindHoles (true);
        rf_est.setRadiusSearch (_rf_rad_);

        rf_est.setInputCloud (_model_keypoints);
        rf_est.setInputNormals (_model_normals);
        rf_est.setSearchSurface (_model);
        rf_est.compute (*model_rf);

        rf_est.setInputCloud (_scene_keypoints);
        rf_est.setInputNormals (_scene_normals);
        rf_est.setSearchSurface (_scene);
        rf_est.compute (*scene_rf);

        //  Clustering
        pcl::Hough3DGrouping<PointT, PointT, RFType, RFType> clusterer;
        clusterer.setHoughBinSize (_cg_size_);
        clusterer.setHoughThreshold (_cg_thresh_);
        clusterer.setUseInterpolation (true);
        clusterer.setUseDistanceWeight (false);

        clusterer.setInputCloud (_model_keypoints);
        clusterer.setInputRf (model_rf);
        clusterer.setSceneCloud (_scene_keypoints);
        clusterer.setSceneRf (scene_rf);
        clusterer.setModelSceneCorrespondences (model_scene_corrs);

        //clusterer.cluster (clustered_corrs);
        clusterer.recognize (rototranslations, clustered_corrs);
    }
    else // Using GeometricConsistency
    {
        pcl::GeometricConsistencyGrouping<PointT, PointT> gc_clusterer;
        gc_clusterer.setGCSize (_cg_size_);
        gc_clusterer.setGCThreshold (_cg_thresh_);

        gc_clusterer.setInputCloud (_model_keypoints);
        gc_clusterer.setSceneCloud (_scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

        //gc_clusterer.cluster (clustered_corrs);
        gc_clusterer.recognize (rototranslations, clustered_corrs);
    }

    //
    //  Output results
    //
    std::cout << "Model instances found: " << rototranslations.size () << std::endl;
    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
    }

    //
    //  Visualization
    //
    //pcl::visualization::PCLVisualizer viewer ("Correspondence Grouping");
    _pcWidget->getPCLVisualizer()->removePointCloud("scene_cloud");
    _pcWidget->getPCLVisualizer()->addPointCloud (_scene, "scene_cloud");

    pcl::PointCloud<PointT>::Ptr off_scene_model (new pcl::PointCloud<PointT> ());
    pcl::PointCloud<PointT>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointT> ());

    if (true)//show_correspondences_ || show_keypoints_)
    {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        pcl::transformPointCloud (*_model, *off_scene_model, Eigen::Vector3f (100,500,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*_model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (100,500,0), Eigen::Quaternionf (1, 0, 0, 0));

        pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        _pcWidget->getPCLVisualizer()->removePointCloud("off_scene_model");
        _pcWidget->getPCLVisualizer()->addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model");
    }

    if (true)//show_keypoints_)
    {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> scene_keypoints_color_handler (_scene_keypoints, 0, 0, 255);
        _pcWidget->getPCLVisualizer()->removePointCloud("scene_keypoints");
        _pcWidget->getPCLVisualizer()->addPointCloud (_scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

        pcl::visualization::PointCloudColorHandlerCustom<PointT> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 255, 0);
        _pcWidget->getPCLVisualizer()->removePointCloud("off_scene_model_keypoints");
        _pcWidget->getPCLVisualizer()->addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
        _pcWidget->getPCLVisualizer()->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
    }

    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointT>::Ptr rotated_model (new pcl::PointCloud<PointT> ());
        pcl::transformPointCloud (*_model, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointT> rotated_model_color_handler (rotated_model, 255, 0, 0);
        _pcWidget->getPCLVisualizer()->removePointCloud(ss_cloud.str ());
        _pcWidget->getPCLVisualizer()->addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

        if (true)//show_correspondences_)
        {
            for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
            {
                std::stringstream ss_line;
                ss_line << "correspondence_line" << i << "_" << j;
                PointT& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
                PointT& scene_point = _scene_keypoints->at (clustered_corrs[i][j].index_match);

                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
                _pcWidget->getPCLVisualizer()->removePointCloud(ss_line.str ());
                _pcWidget->getPCLVisualizer()->addLine<PointT, PointT> (model_point, scene_point, 255, 0, 0, ss_line.str ());
            }
        }
    }
}

Mat_<float> Controller::preOrder(QDomElement node)
{
    cv::Mat_<float> result;
    // operators
    if ( node.tagName() == ("STM2"))
    {
        //qDebug() << "STM2***";
        //qDebug() << node.firstChildElement().tagName();
        if ( node.firstChildElement().text() == ("-") )
        {
            //qDebug() << "MatSub";
            cv::Mat arg1 = preOrder(node.firstChildElement().nextSiblingElement());
            cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement().nextSiblingElement());
            result = arg1 - arg2;
            return result;
        }
        if ( node.firstChildElement().text() == ("+") )
        {
            //qDebug() << "MatAdd";
            cv::Mat arg1 = preOrder(node.firstChildElement().nextSiblingElement());
            cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement().nextSiblingElement());
            result = arg1 + arg2;
            return result;
        }
        if ( node.firstChildElement().text() == ("*") )
        {
            //qDebug() << "MatMul";
            cv::Mat arg1 = preOrder(node.firstChildElement().nextSiblingElement());
            cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement().nextSiblingElement());
            result =  arg1.mul(arg2);
            return result;
        }
    }
    if (node.tagName() == ("STM"))
    {
        //qDebug() << "STM";
        result = preOrder(node.firstChildElement());
        return result;
    }
    if (node.tagName() == ("CONST"))
    {
        //qDebug() << "CONST";
        float cst = node.text().toFloat();//double cst = node.
        cst = cst / 100;
        cv::Mat_<float> cstMat(1, _features[0].cols, cst);
        //assert(_cols == cstMat.cols);
        return cstMat;
    }
    if (node.tagName() == ("VAR"))
    {
        QString strVar = node.text();
        //qDebug() << "******" << strVar;
        // operants
        //qDebug() << "v_" << strVar.mid(2);
        int f_index = strVar.mid(2).toInt();
        return _features[f_index];
    }


    //******************Beagle XML******************

    // operants
    if ( node.tagName() == ("xx") )
    {
        return _features[12];
    }
    if ( node.tagName() == ("xy") )
    {
        return _features[13];
    }
    if ( node.tagName() == ("xz") )
    {
        return _features[14];
    }
    if ( node.tagName() == ("yy") )
    {
        return _features[15];
    }
    if ( node.tagName() == ("yz") )
    {
        return _features[16];
    }
    if ( node.tagName() == ("zz") )
    {
        return _features[17];
    }
    // operators
    if ( node.tagName() == ("MatSub") )
    {
        qDebug() << "MatSub";
        cv::Mat arg1 = preOrder(node.firstChildElement());
        cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement());
        result = arg1 - arg2;
        return result;
    }
    if ( node.tagName() == ("MatAdd") )
    {
        qDebug() << "MatAdd";
        cv::Mat arg1 = preOrder(node.firstChildElement() );
        cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement());
        result = arg1 + arg2;
        return result;
    }
    if ( node.tagName() == ("MatMul") )
    {
        qDebug() << "MatMul";
        cv::Mat arg1 = preOrder(node.firstChildElement() );
        cv::Mat arg2 = preOrder(node.firstChildElement().nextSiblingElement());
        result =  arg1.mul(arg2);
        return result;
    }
    if ( node.tagName() == ("MatMulDou") )
    {
        qDebug() << "MatMulDou";
        cv::Mat arg1 = preOrder(node.firstChildElement() );
        double d = node.firstChildElement().nextSiblingElement().attribute("value").toDouble();
        result =  arg1*d;
        return result;
    }
    if ( node.tagName() == ("MatSquare") )
    {
        qDebug() << "MatSquare";
        cv::Mat arg1 = preOrder(node.firstChildElement());
        result = arg1.mul(arg1);
        return result;
    }
    qDebug() << "Error";
}

Controller::~Controller()
{

}


void Controller::setModel(std::string filename)
{
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool file_is_pcd = true;

    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)	{
        file_is_pcd = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    // Loading file | Works with PCD and PLY files
    pcl::PCLPointCloud2 cloud2;
    if (file_is_pcd) {

        pcl::PCDReader readerPCD;
        readerPCD.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_model);

        std::cout << "Loaded "
                  << _model->width * _model->height
                  << " model from "
                  << filename
                  << std::endl;

        //_pcWidget->getPCLVisualizer()->removePointCloud("model");
        //_pcWidget->getPCLVisualizer()->addPointCloud( _model,"model");

    } else {

        pcl::PLYReader readerPLY;
        readerPLY.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_model);

        std::cout << "Loaded "
                  << _model->width * _model->height
                  << " model from "
                  << filename
                  << std::endl;

        //_pcWidget->getPCLVisualizer()->removePointCloud("model");
        //_pcWidget->getPCLVisualizer()->addPointCloud( _model,"model");
    }
}

void Controller::setScene(std::string filename)
{
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    bool file_is_pcd = true;

    int result = filename.find("pcd");
    std::cout << "find pcd index : " << result << std::endl;
    if (result < 0)
    {
        file_is_pcd = false;
        std::cout << "this is ply file!" << std::endl;
    }
    else
    {
        std::cout << "this is pcd file!" << std::endl;
    }

    // Loading file | Works with PCD and PLY files
    pcl::PCLPointCloud2 cloud2;
    if (file_is_pcd) {

        pcl::PCDReader readerPCD;
        readerPCD.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_scene);

        std::cout << "Loaded "
                  << _scene->width * _scene->height
                  << " scene from "
                  << filename
                  << std::endl;

        //_pcWidget->getPCLVisualizer()->removePointCloud("scene");
        //_pcWidget->getPCLVisualizer()->addPointCloud( _scene,"scene");

    } else {

        pcl::PLYReader readerPLY;
        readerPLY.read (filename, cloud2);
        pcl::fromPCLPointCloud2(cloud2, *_scene);

        std::cout << "Loaded "
                  << _scene->width * _scene->height
                  << " scene from "
                  << filename
                  << std::endl;

        //_pcWidget->getPCLVisualizer()->removePointCloud("scene");
        //_pcWidget->getPCLVisualizer()->addPointCloud( _scene,"scene");
    }
}
