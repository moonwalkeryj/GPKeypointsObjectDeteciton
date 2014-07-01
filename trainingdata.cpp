#include "trainingdata.h"
#include <pcl/features/normal_3d.h>

void TrainingData::saveTrainingFeaturesAndTarget(QString filename)
{

    for (int i = 0; i < 18; i++)
    {
        cv::FileStorage file( (filename+QString::number(i)+".xml").toStdString(), cv::FileStorage::WRITE );
        file << "f" << _features[i]; file.release();
        std::cout << "file: " <<  (filename+QString::number(i)+".xml").toStdString() << " saved!" << std::endl;
    }
    cv::FileStorage file( (filename+"_target.xml").toStdString(), cv::FileStorage::WRITE );
    file << "f" << _target; file.release();
    std::cout << "file: " <<  (filename+"_target.xml").toStdString() << " saved!" << std::endl;
}


void TrainingData::setTarget(QString filename)
{
    cv::Mat temp;
    //populate desired corner mask
    cv::FileStorage storagemsk(filename.toAscii().data(), cv::FileStorage::READ); storagemsk["f"] >> temp; storagemsk.release();
    std::cout << "File Opened Info[ rows: " << temp.rows << "; cols: " << temp.cols << " ]"<< std::endl;
    //_target = temp.clone();
    _targetT = temp.clone();
    std::cout << "Target file info[ rows: " << _targetT.rows << "; cols: " << _targetT.cols << " ]"<< std::endl;
}


TrainingData::TrainingData(QObject *parent)
{

}


TrainingData::~TrainingData()
{

}


void TrainingData::extractFeatrues(CloudTPtr &cloud, cv::Mat_<float> *features, float search_radius)
{
    clock_t begin, end;
    double time_spent;

    begin = clock();
    /* here, do your time-consuming job */

    if (cloud->empty())
    {
        std::cout << "no cloud selected..." << std::endl;
    }

    // Calculate Normals
    PointCloudNConstPtr normals_;
    PointCloudNPtr normals (new PointCloudN ());
    normals->reserve (normals->size ());
    pcl::NormalEstimation<PointT, NormalT> normal_estimation;
    normal_estimation.setInputCloud (cloud);
    //normal_estimation.setKSearch(8); //***********
    //normal_estimation.setRadiusSearch (search_radius_);
    normal_estimation.setRadiusSearch (search_radius);
    normal_estimation.compute (*normals);
    normals_ = normals;

    int num_col = cloud->size();
    //cv::Mat_<float> _f[24];
    for (int i = 0; i < 18; i++)
    {
        std::cout << "--initialize feature mat[" << i << "]: is continuous ? ";
        features[i] = cv::Mat_<float>(1, num_col);
        if (features[i].isContinuous()) std::cout << "yes!--" << std::endl;
        else std::cout << "no!--" << std::endl;
    }

    pcl::KdTreeFLANN<PointT>::Ptr tree_(new pcl::KdTreeFLANN<PointT>());

    tree_->setInputCloud (cloud);

    for (int pIdx = 0; pIdx < static_cast<int> (cloud->size ()); ++pIdx)
    {
        const PointT& pointIn = cloud->points [pIdx];

        if (isFinite (pointIn))
        {
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;

            // n-nearest number search
            //tree_->nearestKSearch(pointIn, 8, nn_indices, nn_dists);
            // radius search
            //float search_radius_ = 0.15;
            tree_->radiusSearch (pointIn, search_radius, nn_indices, nn_dists);

            // Terminal set2
            unsigned count = 0;
            float statistics[12];
            // calculate 0 ~ 11
            // avg 0 1 2; std 3 4 5; max 6 7 8; min 9 10 11;
            memset (statistics, 0, sizeof (float) * 12);

            statistics[6]  = std::numeric_limits<float>::min();
            statistics[7]  = std::numeric_limits<float>::min();
            statistics[8]  = std::numeric_limits<float>::min();

            statistics[9]  = std::numeric_limits<float>::max();
            statistics[10] = std::numeric_limits<float>::max();
            statistics[11] = std::numeric_limits<float>::max();


            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {
                if (pcl_isfinite (normals_->points[*iIt].normal_x))
                {
                    statistics[0] += normals_->points[*iIt].normal_x;
                    statistics[1] += normals_->points[*iIt].normal_y;
                    statistics[2] += normals_->points[*iIt].normal_z;

                    if ( statistics[6]  < normals_->points[*iIt].normal_x ) statistics[6]  = normals_->points[*iIt].normal_x;
                    if ( statistics[7]  < normals_->points[*iIt].normal_y ) statistics[7]  = normals_->points[*iIt].normal_y;
                    if ( statistics[8]  < normals_->points[*iIt].normal_z ) statistics[8]  = normals_->points[*iIt].normal_z;

                    if ( statistics[9]  > normals_->points[*iIt].normal_x ) statistics[9]  = normals_->points[*iIt].normal_x;
                    if ( statistics[10] > normals_->points[*iIt].normal_y ) statistics[10] = normals_->points[*iIt].normal_y;
                    if ( statistics[11] > normals_->points[*iIt].normal_z ) statistics[11] = normals_->points[*iIt].normal_z;

                    ++count;
                }
            }

            float norm;
            if (count > 0)
            {
                norm = 1.0 / float (count);
                statistics[0] *= norm;
                statistics[1] *= norm;
                statistics[2] *= norm;
            }

            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {
                if (pcl_isfinite (normals_->points[*iIt].normal_x))
                {
                    statistics[3] += sqrt((normals_->points[*iIt].normal_x - statistics[0])*(normals_->points[*iIt].normal_x - statistics[0]));
                    statistics[4] += sqrt((normals_->points[*iIt].normal_y - statistics[1])*(normals_->points[*iIt].normal_y - statistics[1]));
                    statistics[5] += sqrt((normals_->points[*iIt].normal_z - statistics[2])*(normals_->points[*iIt].normal_z - statistics[2]));
                }
            }

            if (count > 0)
            {
                norm = 1.0 / float (count);
                statistics[3] *= norm;
                statistics[4] *= norm;
                statistics[5] *= norm;
            }

            // save 0 ~ 11
            for (int i = 0; i < 12; i++)
            {
                ((float*)features[i].data)[pIdx] = statistics[i];
            }

            // calculate 12 ~ 17
            float coefficients[6];
            memset (coefficients, 0, sizeof (float) * 6);
            count = 0;
            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {
                if (pcl_isfinite (normals_->points[*iIt].normal_x))
                {
                    coefficients[0] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_x;
                    coefficients[1] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_y;
                    coefficients[2] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_z;

                    coefficients[3] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_y;
                    coefficients[4] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_z;
                    coefficients[5] += normals_->points[*iIt].normal_z * normals_->points[*iIt].normal_z;

                    ++count;
                }
            }

            if (count > 0)
            {
                float norm = 1.0 / float (count);
                coefficients[0] *= norm;
                coefficients[1] *= norm;
                coefficients[2] *= norm;
                coefficients[3] *= norm;
                coefficients[4] *= norm;
                coefficients[5] *= norm;
            }
            // save 12 ~ 17
            for (int i = 0; i < 6; i++)
            {
                ((float*)features[i+12].data)[pIdx] = coefficients[i];
            }

            // Terminal set1
            //            int normal_index = 0;
            //            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt, ++normal_index)
            //            {
            //                if (pcl_isfinite (normals_->points[*iIt].normal_x))
            //                {
            //                    ((float*)features[3*normal_index + 0].data)[pIdx] = normals_->points[*iIt].normal_x;
            //                    ((float*)features[3*normal_index + 1].data)[pIdx] = normals_->points[*iIt].normal_y;
            //                    ((float*)features[3*normal_index + 2].data)[pIdx] = normals_->points[*iIt].normal_z;
            //                }
            //            }
        }
    }

    // output calculating time
    std::cout << "Feature Extracted...." << std::endl;
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Processing Time: " << time_spent << "s;" << std::endl;
}


void TrainingData::appendFeatruesAndObjective(cv::Mat_<float> *dstfeatures, cv::Mat_<uchar>& dsttarget, cv::Mat_<float> *srcfeatures, cv::Mat_<uchar>& srctarget)
{
    // append training data's features
    std::cout << "******--------" << std::endl;
    for (int i = 0; i < 18; i++)
    {
        std::cout << "Before: rows: " << dstfeatures[i].rows << "; cols: " << dstfeatures[i].cols << std::endl;

        dstfeatures[i].push_back(srcfeatures[i].clone());
        std::cout << "After: rows: " << dstfeatures[i].rows << "; cols: " << dstfeatures[i].cols << std::endl;

        if (dstfeatures[i].isContinuous()) std::cout << "Mat feature is continuous !!" << i << std::endl;
        else std::cout << "Mat feature is not continuous" << std::endl;
    }

    // append target
    dsttarget.push_back(srctarget.clone());
    std::cout << "Target File Updated Info [ rows: " << dsttarget.rows << "; cols: " << dsttarget.cols << " ]"<< std::endl;
    if (dsttarget.isContinuous()) std::cout << "Target Mat is continuous !!" << std::endl;
    else std::cout << "Target Mat is not continuous" << std::endl;
}


void TrainingData::appendFeaturesAndTarget(TrainingData::CloudTPtr &cloud)
{
    cv::Mat_<float> features[18];
    extractFeatrues(cloud, features);
    appendFeatruesAndObjective(_features, _target, features, _targetT);
}

