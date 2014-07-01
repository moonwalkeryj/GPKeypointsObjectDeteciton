#ifndef CLOUDPROCESSORIMP_H
#define CLOUDPROCESSORIMP_H
#include "cloudprocessor.h"

template<typename PointT>
CloudProcessor<PointT>::CloudProcessor() :
    _voxelSize(0.06),
    _voxelSizeSelfUse(0.06),
    _isGrdUpdateThreadRunning(false),
    _DownSampler(new pcl::VoxelGrid<PointT>),
    _DownSamplerSelfUse(new pcl::VoxelGrid<PointT>),
    _PlaneDetSegmentor(new pcl::SACSegmentation<PointT>),
    _planeDistanceThreshold(0.03),
    _IndicesExtractor(new pcl::ExtractIndices<PointT>),
    _IndicesExtractorSelfUse(new pcl::ExtractIndices<PointT>),
    _PlaneModelExtractor(new pcl::SampleConsensusModelPlane<PointT>(boost::shared_ptr<pcl::PointCloud<PointT>>(new pcl::PointCloud<PointT>))),
    _BoundingBoxModels(),
    _mutex_boundingboxes()
{
    // Ground Coefficients
    _groundCoeffs.resize(4);
    _groundCoeffs[0] = -0.0244234;
    _groundCoeffs[1] = -0.998427;
    _groundCoeffs[2] = -0.0504705;
    _groundCoeffs[3] = 0.764362;

    // down sampler
    _DownSampler->setLeafSize (_voxelSize, _voxelSize, _voxelSize);
    _DownSamplerSelfUse->setLeafSize (_voxelSize, _voxelSize, _voxelSize);

    // Plane Segmentator
    _PlaneDetSegmentor->setOptimizeCoefficients (true);
    _PlaneDetSegmentor->setModelType (pcl::SACMODEL_PLANE);
    _PlaneDetSegmentor->setMethodType (pcl::SAC_RANSAC);
    _PlaneDetSegmentor->setMaxIterations (1000);
    _PlaneDetSegmentor->setDistanceThreshold (_planeDistanceThreshold);
}

template<typename PointT>
CloudProcessor<PointT>::~CloudProcessor()
{

}

template<typename PointT>
void CloudProcessor<PointT>::groundCoeffUpdateThread()
{


    FPS_CALC("Thread, Ground Coeff Updating");
    //  only one thread is running
    if (_isGrdUpdateThreadRunning) return;

    _isGrdUpdateThreadRunning = true;
    boost::shared_ptr<pcl::PointCloud<PointT>> down_sampled_cloud(new pcl::PointCloud<PointT>);

    // Downsample of sampling_factor in every dimension:
    /*
    int sampling_factor_ = 1;
    if (sampling_factor_ != 1)
    {
        CloudTPtr cloud_downsampled(new CloudT);
        cloud_downsampled->width = (input_->width)/sampling_factor_;
        cloud_downsampled->height = (input_->height)/sampling_factor_;
        cloud_downsampled->points.resize(cloud_downsampled->height*cloud_downsampled->width);
        cloud_downsampled->is_dense = input_->is_dense;
        for (int j = 0; j < cloud_downsampled->width; j++)
        {
            for (int i = 0; i < cloud_downsampled->height; i++)
            {
                (*cloud_downsampled)(j,i) = (*input_)(sampling_factor_*j,sampling_factor_*i);
            }
        }
        (*down_sampled_cloud) = (*cloud_downsampled);
    }
    */

    // Voxel grid down sampling:
    _DownSamplerSelfUse->setInputCloud(input_);
    _DownSamplerSelfUse->filter (*down_sampled_cloud);

    if (down_sampled_cloud->size() < 3) return;

    // Planes detection and segmentation
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    int iterations = 1;
    float * models = new float[4*iterations];
    boost::shared_ptr<pcl::PointCloud<PointT>> no_plane_cloud(new pcl::PointCloud<PointT>);
    for (int i = 0; i < iterations; i++) {
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
        inliers->indices.clear();
        // detect plane and segment
        _PlaneDetSegmentor->setInputCloud (down_sampled_cloud);
        _PlaneDetSegmentor->segment (*inliers, *coefficients);
        // extract no plane cloud
        _IndicesExtractorSelfUse->setInputCloud(down_sampled_cloud);
        _IndicesExtractorSelfUse->setIndices(inliers);
        _IndicesExtractorSelfUse->setNegative(true);
        _IndicesExtractorSelfUse->filter(*no_plane_cloud);
        down_sampled_cloud.swap( no_plane_cloud );
        // populate each members for model i
        for(int j = 0; j < 4; j++)
        {
            models[i*4+j] = coefficients->values[j];
        }
    }


    // Clustering
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(down_sampled_cloud);
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(2 * 0.06);
    int min_points_ = 200;     // this value is adapted to the voxel size in method "compute"
    int max_points_ = 5000;   // this value is adapted to the voxel size in method "compute"
    ec.setMinClusterSize(min_points_);
    ec.setMaxClusterSize(max_points_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(down_sampled_cloud);
    ec.extract(cluster_indices);
    //std::cerr << "clusters size:" << cluster_indices.size() << std::endl;



    // clear _bounding box models vector
    _mutex_boundingboxes.lock();
    _BoundingBoxModels.clear();


    for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        //std::cerr << (*it).indices.size() << std::endl;

        float   min_x, min_y, min_z,
                max_x, max_y, max_z,
                sum_x, sum_y, sum_z,
                centroid_x, centroid_y, centroid_z;
        min_x = min_y = min_z = 5.0;
        max_x = max_y = max_z = -5.0;
        sum_x = sum_y = sum_z = 0;

        int size = 0;

        for(std::vector<int>::const_iterator it2 = (*it).indices.begin(); it2 != (*it).indices.end(); ++it2)
        {
            PointT * p = &down_sampled_cloud->points[*it2];

            min_x = std::min(p->x, min_x);
            max_x = std::max(p->x, max_x);
            sum_x += p->x;

            min_y = std::min(p->y, min_y);
            max_y = std::max(p->y, max_y);
            sum_y += p->y;

            min_z = std::min(p->z, min_z);
            max_z = std::max(p->z, max_z);
            sum_z += p->z;

            size++;
        }

        centroid_x = sum_x / size;
        centroid_y = sum_y / size;
        centroid_z = sum_z / size;

//        std::cerr << "clustered cloud:"
//                  << min_x << "," << min_y << "," << min_z
//                  << max_x << "," << max_y << "," << max_z
//                  << sum_x << "," << sum_y << "," << sum_z
//                  << centroid_x << "," << centroid_y << "," << centroid_z
//                  << std::endl;

        // Bounding Box
        // Coeff
        pcl::ModelCoefficients coeffs;
        // translation: the centroid of the object
        coeffs.values.push_back ((max_x + min_x) / 2); //coeffs.values.push_back (centroid_x);
        coeffs.values.push_back ((max_y + min_y) / 2); //coeffs.values.push_back (centroid_y);
        coeffs.values.push_back ((max_z + min_z) / 2); //coeffs.values.push_back (centroid_z);
        // rotation
        coeffs.values.push_back (_groundCoeffs[0]);
        coeffs.values.push_back (_groundCoeffs[1]);
        coeffs.values.push_back (_groundCoeffs[2]);
        coeffs.values.push_back (1.0);
        // size Vertical
        coeffs.values.push_back (max_x - min_x); // x
        coeffs.values.push_back (max_y - min_y); // y
        coeffs.values.push_back (max_z - min_z); // z
        // push back
        _BoundingBoxModels.push_back(coeffs);
    }

    // Bounding Box mutex unlock
    _mutex_boundingboxes.unlock();


    // find the ground and update
    int smallestAnglePos = 0;
//    float * angles = new float[iterations];
//    float smallestAngle = 1.5;
    float angleThreshold = 0.5;
    std::vector<int> filteredAngleIndex;// angle that is less than the threshold// now not consider angle lager than 90 degree
    for(int i = 0; i < iterations; i++)
    {
        // arccos(angle) = A*B / |A|*|B|
        float a = std::sqrt(models[i*4]*models[i*4] + models[i*4+1]*models[i*4+1] + models[i*4+2]*models[i*4+2]);
        float b = std::sqrt(_groundCoeffs[0]*_groundCoeffs[0] + _groundCoeffs[1]*_groundCoeffs[1] + _groundCoeffs[2]*_groundCoeffs[2]);
        float temp = (models[i*4]*_groundCoeffs[0] + models[i*4+1]*_groundCoeffs[1] + models[i*4+2]*_groundCoeffs[2])/(a*b);

        //angles[i] = std::acos(temp/(a*b));
        float angle = std::acos(temp/(a*b));

        if(/*angles[i]*/ angle < angleThreshold)
        {
            filteredAngleIndex.push_back(i);
        }
//        if (angles[i] < smallestAngle )
//        {
//            smallestAnglePos = i;
//            smallestAngle = angles[i];
//        }
    }

    // filter modles with the smallest d
    bool first = true;
    int smallestZIndex = 0;

    for(std::vector<int>::const_iterator it = filteredAngleIndex.begin();
            it != filteredAngleIndex.end(); it++)
    {
        if(first) {smallestZIndex = *it; first = false; continue;}
        if(models[(*it)*4+3] < models[smallestZIndex*4+3])
        {
            smallestZIndex = *it;
        }
    }

    // rewrite the ground coefficients
    std::stringstream s;
    s << "---Thread---Coord Coeffecients:";
    _mutexGndCef.lock();
    for(int i = 0; i < 4; i++)
    {
        _groundCoeffs[i] = models[smallestZIndex*4 + i];
        s << models[smallestZIndex*4 + i] << " ";
    }
    _mutexGndCef.unlock();
    //std::cerr << s.str() << std::endl;
    _isGrdUpdateThreadRunning = false;

    // release models & angles

    delete [] models;
//    delete [] angles;
}

template<typename PointT>
float CloudProcessor<PointT>::getVoxelSize() const
{
    return _voxelSize;
}

template<typename PointT>
void CloudProcessor<PointT>::setVoxelSize(float value)
{
    _voxelSize = value;
}

template<typename PointT>
Eigen::VectorXf CloudProcessor<PointT>::getGroundCoeffs()
{
    //_mutexGndCef.lock();
    Eigen::VectorXf temp = _groundCoeffs;
    //_mutexGndCef.unlock();
    return temp;
}

template<typename PointT>
void CloudProcessor<PointT>::setGroundCoeffs(const Eigen::VectorXf &value)
{
    _groundCoeffs = value;
}

template<typename PointT>
void CloudProcessor<PointT>::setGroundCoeffs(float a, float b, float c, float d){
    _groundCoeffs[0] = a;
    _groundCoeffs[1] = b;
    _groundCoeffs[2] = c;
    _groundCoeffs[3] = d;
}

template<typename PointT>
void CloudProcessor<PointT>::setGroundCoeffs(float x, PLANECOEFF COEFF)
{
    _groundCoeffs[COEFF] = x;
}

template<typename PointT>
bool CloudProcessor<PointT>::isGroundCoeffUpdating() const
{
    return _isGrdUpdateThreadRunning;
}

#define PCL_INSTANTIATE_CloudProcessor(T) template class PCL_EXPORTS CloudProcessor<T>;
#endif
