#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <pcl/keypoints/keypoint.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include "qpointcloudwidget.h"
#include <QtXml>
#include <QObject>


class ObjectDetector : public QObject{

Q_OBJECT
private:
    typedef pcl::PointXYZRGB                PointT;
    typedef pcl::PointCloud <PointT>        CloudT;
    typedef CloudT::Ptr                     CloudTPtr;
    typedef CloudT::ConstPtr                CloudTConstPtr;

    typedef pcl::PointXYZRGBNormal              PointXYZRGBNormal;
    typedef pcl::PointCloud <PointXYZRGBNormal> CloudXYZRGBNormal;
    typedef CloudXYZRGBNormal::Ptr              CloudXYZRGBNormalPtr;
    typedef CloudXYZRGBNormal::ConstPtr         CloudXYZRGBNormalConstPtr;

    typedef pcl::Normal NormalT;

    typedef pcl::SHOT352 DescriptorType;

    typedef  pcl::PointCloud<NormalT> PointCloudN;
    typedef  PointCloudN::Ptr PointCloudNPtr;
    typedef  PointCloudN::ConstPtr PointCloudNConstPtr;


public slots:

public:

    ObjectDetector(QPointCloudWidget * parent);
    ~ObjectDetector();
private:
    pcl::PointCloud<PointT>::Ptr _model;
    pcl::PointCloud<PointT>::Ptr _model_keypoints;
    pcl::PointCloud<PointT>::Ptr _scene;
    pcl::PointCloud<PointT>::Ptr _scene_keypoints;
    pcl::PointCloud<NormalT>::Ptr _model_normals;
    pcl::PointCloud<NormalT>::Ptr _scene_normals;
    pcl::PointCloud<DescriptorType>::Ptr _model_descriptors;
    pcl::PointCloud<DescriptorType>::Ptr _scene_descriptors;

    float _model_ss_;
    float _scene_ss_;
    float _rf_rad_;
    float _descr_rad_;
    float _cg_size_;
    float _cg_thresh_;

    bool _show_keypoints_;
    bool _show_correspondences_;
    bool _use_cloud_resolution_;
    bool _use_hough_;

    /** \brief Widget to Display Point Cloud*/
    QPointCloudWidget * _pcWidget;
    /** \brief features*/
    cv::Mat_<float> _features[24];

public slots:

    // Set Model
    void setModel(std::string filename);

    // Set Scene
    void setScene(std::string filename);

    // Initialize Resolution
    void initModelAndSceneResolution();  // ****************

    // detect keypoints of model
    void detectModelGPKeypoints(QString strGeno);

    // detect keypoints of scene
    void detectSceneGPKeypoints(QString strGeno);

    // compute descriptors of model
    void computeModelDescriptors();

    // compute descriptors of scene
    void computeSceneDescriptors();

    // Detect key points
    cv::Mat_<float> preOrder(QDomElement node);

public:
    double computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud);

};

#endif
