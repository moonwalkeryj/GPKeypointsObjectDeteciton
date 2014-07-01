
#ifndef CLOUDPROCESSOR_H
#define CLOUDPROCESSOR_H
#include <boost/thread/mutex.hpp>


#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/geometry/triangle_mesh.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/filters/extract_indices.h>

// Useful macros
#define FPS_CALC(_WHAT_) \
do \
{ \
    static unsigned count = 0;\
    static double last = pcl::getTime ();\
    double now = pcl::getTime (); \
    ++count; \
    if (now - last >= 1.0) \
    { \
      std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(now - last) << " Hz" <<  std::endl; \
      count = 0; \
      last = now; \
    } \
}while(false)

template<typename PointT>
class CloudProcessor : public pcl::PCLBase<PointT>{

    using PCLBase<PointT>::input_;

    class StopWatch{
    private:
        double old;
    public:
        StopWatch():old(0.0){
        }
        inline void initial(){old = pcl::getTime();}
        // show computation time and recount
        inline void show(std::string title){
            std::cout << "StopWatch-Computation Time of "<< title << " : " << pcl::getTime() - old << "s" << std::endl;
            old = pcl::getTime();
        }
    };


private:
    // down sampling voxel size
    float _voxelSize;
    float _voxelSizeSelfUse;
    // Plane detect and seg distance threshold
    float _planeDistanceThreshold;

    // Ground Coefficients:
    Eigen::VectorXf _groundCoeffs;
    // is Ground Update Thread running
    bool _isGrdUpdateThreadRunning;


public:
    CloudProcessor();
    ~CloudProcessor();

    void process();

    void groundCoeffUpdateThread();

    // Voxel grid downsampler
    boost::shared_ptr<pcl::VoxelGrid<PointT>> _DownSampler;

    // Voxel grid downsampler
    boost::shared_ptr<pcl::VoxelGrid<PointT>> _DownSamplerSelfUse;

    // Plane detector and segmentor
    boost::shared_ptr<pcl::SACSegmentation<PointT>> _PlaneDetSegmentor;

    // Cloud Extractor
    boost::shared_ptr<pcl::ExtractIndices<PointT>> _IndicesExtractor;

    // Cloud Extractor
    boost::shared_ptr<pcl::ExtractIndices<PointT>> _IndicesExtractorSelfUse;

    // Plane Extractor with Plane Model
    boost::shared_ptr<pcl::SampleConsensusModelPlane<PointT>> _PlaneModelExtractor;

    // Vector for Bounding Box Models
    std::vector<pcl::ModelCoefficients> _BoundingBoxModels;

    // Mutex for Bounding Box Models
    boost::mutex _mutex_boundingboxes;
    // mutex
    boost::mutex _mutexGndCef;

    // Voxel size getter & setter
    float getVoxelSize() const;
    void setVoxelSize(float value);

    // Ground Coeff getter & setter
    Eigen::VectorXf getGroundCoeffs();
    void setGroundCoeffs(const Eigen::VectorXf &value);
    void setGroundCoeffs(float a, float b, float c, float d);
    // These are the plane coefficients for the plane
    // Definition: ax + by + cz + d=0
    enum PLANECOEFF {A=0, B, C, D} ;
    void setGroundCoeffs(float x, PLANECOEFF COEFF);

    // getter of isGround updating
    bool isGroundCoeffUpdating() const;
};
#endif
