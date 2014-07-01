#ifndef CONTROLLER_H
#define CONTROLLER_H


#ifndef Q_MOC_RUN
#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include "qpointcloudwidget.h"
#include "cloudprocessor.h"
#include <opencv2/opencv.hpp>
#include <pcl/keypoints/keypoint.h>
#include <QtXml>
#endif

#include "trainingdata.h"
#include <QObject>

class Controller : public QObject{

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

    typedef pcl::ReferenceFrame RFType;

    typedef pcl::SHOT352 DescriptorType;
//    typedef pcl::PFHSignature125 DescriptorType;

    typedef  pcl::PointCloud<NormalT> PointCloudN;
    typedef  PointCloudN::Ptr PointCloudNPtr;
    typedef  PointCloudN::ConstPtr PointCloudNConstPtr;

	typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> ColorHandlerT3;
    typedef pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ColorHandlerXYZ;

public slots:
    // reset camera view
    void resetCameraView();

    // Set Model
    void showModel(std::string filename);

    // Set Scene
    void showScene(std::string filename);

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

    // append training features
    void appendTrainingFeatures();

    // Correpondence Computing
    //void corresponce();


    // slots to show point cloud
    void toggleShowHarrisKeypoints(bool ischecked);
    void toggleShowISSKeypoints(bool ischecked);
    void toggleShowGPKeypoints(bool ischecked, QString strGeno);

    // set gp threshold
    void setGpThreshold(int gpThreshold);

    // save point cloud
    void savePointCloud(QString filename);

    // save keypoints
    void saveKeypoints(QString filename);

    // rotate current point cloud
    void rotateX();
    void rotate(int degrees, int axis);
    void translate(int distance, int axis);

    // object detection test
    void objectDetectionTest();

public:
    Controller(QPointCloudWidget * pointcloudWidget);
    ~Controller();

    // Display point cloud
    void showPointCloud(std::string filename);

    // Show keypoints
    void showKeypoints(std::string filename);
    void removeKeypoints();

    // HARRIS Keypoint Detector
    void detectHarris3DKeypoints();
    void removeHarris3DKeypoints();

    // ISS Keypoint Detector
    void detectISSKeypoints();
    void removeISSKeypoints();

    // Geno Keypoints Detector
    void detectGPKeypoints(QString strGeno);
    void removeGPKeypoints();

    // Generate artificial point cloud
    void generateArtificialCloud();

    // Set Model
    void setModel(std::string filename);

    // Set Scene
    void setScene(std::string filename);

    // Detect key points
    cv::Mat_<float> preOrder(QDomElement node);

    double computeCloudResolution (const pcl::PointCloud<PointT>::ConstPtr &cloud);

private:
    /** \brief Pointer of Cloud Data for Display. */
    CloudTPtr _cloudPtr;

    /** \brief Pointer of Cloud Data for Display. */
    CloudTPtr _cloudKeypoints;

    /** \brief Synchronization. */
    boost::mutex _mutex;

    /** \brief Is New Data Available or Not*/
    bool _new_data_available;

    /** \brief Widget to Display Point Cloud*/
    QPointCloudWidget * _pcWidget;

    /** \brief Processor Pointer*/
    //boost::shared_ptr<CloudProcessor<PointT>> _processor;

    /** \brief features*/
    cv::Mat_<float> _features[24];

    /** \brief features*/
    cv::Mat_<uchar> _target;

    /** \brief GP detection threshold*/
    int _gpThreshold;

    pcl::PointCloud<PointT>::Ptr _model;
    pcl::PointCloud<PointT>::Ptr _model_keypoints;
    pcl::PointCloud<PointT>::Ptr _scene;
    pcl::PointCloud<PointT>::Ptr _scene_keypoints;
    pcl::PointCloud<NormalT>::Ptr _model_normals;
    pcl::PointCloud<NormalT>::Ptr _scene_normals;
    pcl::PointCloud<DescriptorType>::Ptr _model_descriptors;
    pcl::PointCloud<DescriptorType>::Ptr _scene_descriptors;

//    pcl::PointCloud<PFH>::Ptr _model_descriptors_pfh;
//    pcl::PointCloud<PFH>::Ptr _scene_descriptors_pfh;

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

public:
    /** \brief Training Features*/
    TrainingData _trainingData;

};

#endif
