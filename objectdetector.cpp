#include "objectdetector.h"
#include "trainingdata.h"

#include <QDebug>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot_omp.h>
ObjectDetector::ObjectDetector(QPointCloudWidget *parent):
_pcWidget(parent)
{

}


ObjectDetector::~ObjectDetector()
{

}

void ObjectDetector::setModel(std::string filename)
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

void ObjectDetector::setScene(std::string filename)
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

double ObjectDetector::computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud)
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

void ObjectDetector::initModelAndSceneResolution()
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

void ObjectDetector::detectModelGPKeypoints(QString strGeno)
{
    //
    // Using GP Keypoints Detector
    //
    TrainingData::extractFeatrues(_model, _features);

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
        if (((uchar*)norm.data)[i] > 128)//_gpThreshold)
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

void ObjectDetector::detectSceneGPKeypoints(QString strGeno)
{
    //
    // Using GP Keypoints Detector
    //
    TrainingData::extractFeatrues(_scene, _features);

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
        if (((uchar*)norm.data)[i] > 128)//_gpThreshold)
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

void ObjectDetector::computeModelDescriptors()
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


    pcl::SHOTEstimationOMP<PointT, NormalT, DescriptorType> descr_est;
    descr_est.setRadiusSearch (_descr_rad_);

    descr_est.setInputCloud (_model_keypoints);
    descr_est.setInputNormals (_model_normals);
    descr_est.setSearchSurface (_model);
    descr_est.compute (*_model_descriptors);
}

void ObjectDetector::computeSceneDescriptors()
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


    pcl::SHOTEstimationOMP<PointT, NormalT, DescriptorType> descr_est;
    descr_est.setRadiusSearch (_descr_rad_);

    descr_est.setInputCloud (_scene_keypoints);
    descr_est.setInputNormals (_scene_normals);
    descr_est.setSearchSurface (_scene);
    descr_est.compute (*_scene_descriptors);
}

cv::Mat_<float> ObjectDetector::preOrder(QDomElement node)
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
    qDebug() << "Error";
}
