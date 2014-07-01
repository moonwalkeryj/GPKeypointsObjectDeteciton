#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include <pcl/io/pcd_io.h>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <pcl/keypoints/keypoint.h>

#include <QObject>

class TrainingData : public QObject{

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

    // save training features and target
    void saveTrainingFeaturesAndTarget(QString filename);
    // Set Targets
    void setTarget(QString filename);

public:

    TrainingData(QObject * parent = NULL);
    ~TrainingData();
public:
    // Extract features from selected point cloud
    static void extractFeatrues(CloudTPtr &cloud, cv::Mat_<float>* features, float search_radius = 1.0f);

private:
    // Append new features and training target to current Training set
    static void appendFeatruesAndObjective( cv::Mat_<float>* dstfeatures, cv::Mat_<uchar>& dsttarget, cv::Mat_<float>* srcfeatures, cv::Mat_<uchar>& srctarget);

public:
    // Append new features and training target to current Training set
    void appendFeaturesAndTarget(CloudTPtr &cloud);


public:
    /** \brief features*/
    cv::Mat_<float> _features[18];

    /** \brief target response*/
    cv::Mat_<uchar> _target;

	/** \brief target response*/
    cv::Mat_<uchar> _targetT;
};

#endif
