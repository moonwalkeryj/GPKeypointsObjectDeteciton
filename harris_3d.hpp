/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_GP_KEYPOINT_3D_IMPL_H_
#define PCL_GP_KEYPOINT_3D_IMPL_H_

#include "harris_3d.h"
#include <pcl/common/io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/time.h>
#include <pcl/common/centroid.h>
#include "trainingdata.h"
#ifdef __SSE__
#include <xmmintrin.h>
#endif

using namespace cv;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setMethod (ResponseMethod method)
{
    method_ = method;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setThreshold (float threshold)
{
    threshold_= threshold;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setRadius (float radius)
{
    search_radius_ = radius;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setRefine (bool do_refine)
{
    refine_ = do_refine;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setNonMaxSupression (bool nonmax)
{
    nonmax_ = nonmax;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setNormals (const PointCloudNConstPtr &normals)
{
    normals_ = normals;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::calculateNormalCovar (const std::vector<int>& neighbors, float* coefficients)
{
    unsigned count = 0;
    // indices        0   1   2   3   4   5   6   7
    // coefficients: xx  xy  xz  ??  yx  yy  yz  ??
#ifdef __SSE__
    // accumulator for xx, xy, xz
    __m128 vec1 = _mm_setzero_ps();
    // accumulator for yy, yz, zz
    __m128 vec2 = _mm_setzero_ps();

    __m128 norm1;

    __m128 norm2;

    float zz = 0;

    for (std::vector<int>::const_iterator iIt = neighbors.begin(); iIt != neighbors.end(); ++iIt)
    {
        if (pcl_isfinite (normals_->points[*iIt].normal_x))
        {
            // nx, ny, nz, h
            norm1 = _mm_load_ps (&(normals_->points[*iIt].normal_x));

            // nx, nx, nx, nx
            norm2 = _mm_set1_ps (normals_->points[*iIt].normal_x);

            // nx * nx, nx * ny, nx * nz, nx * h
            norm2 = _mm_mul_ps (norm1, norm2);

            // accumulate
            vec1 = _mm_add_ps (vec1, norm2);

            // ny, ny, ny, ny
            norm2 = _mm_set1_ps (normals_->points[*iIt].normal_y);

            // ny * nx, ny * ny, ny * nz, ny * h
            norm2 = _mm_mul_ps (norm1, norm2);

            // accumulate
            vec2 = _mm_add_ps (vec2, norm2);

            zz += normals_->points[*iIt].normal_z * normals_->points[*iIt].normal_z;
            ++count;
        }
    }
    if (count > 0)
    {
        norm2 = _mm_set1_ps (float(count));
        vec1 = _mm_div_ps (vec1, norm2);
        vec2 = _mm_div_ps (vec2, norm2);
        _mm_store_ps (coefficients, vec1);
        _mm_store_ps (coefficients + 4, vec2);
        coefficients [7] = zz / float(count);
    }
    else
        memset (coefficients, 0, sizeof (float) * 8);
#else
    memset (coefficients, 0, sizeof (float) * 8);
    for (std::vector<int>::const_iterator iIt = neighbors.begin(); iIt != neighbors.end(); ++iIt)
    {
        if (pcl_isfinite (normals_->points[*iIt].normal_x))
        {
            coefficients[0] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_x;
            coefficients[1] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_y;
            coefficients[2] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_z;

            coefficients[5] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_y;
            coefficients[6] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_z;
            coefficients[7] += normals_->points[*iIt].normal_z * normals_->points[*iIt].normal_z;

            ++count;
        }
    }
    if (count > 0)
    {
        float norm = 1.0 / float (count);
        coefficients[0] *= norm;
        coefficients[1] *= norm;
        coefficients[2] *= norm;
        coefficients[5] *= norm;
        coefficients[6] *= norm;
        coefficients[7] *= norm;
    }
#endif
}
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::calculateStatistics(const std::vector<int> &neighbors, float *statistics)
{
    unsigned count = 0;

    // avg 0 1 2; std 3 4 5; max 6 7 8; min 9 10 11;

    memset (coefficients, 0, sizeof (float) * 8);

    statistics[6]  = std::numeric_limits<float>::min();
    statistics[7]  = std::numeric_limits<float>::min();
    statistics[8]  = std::numeric_limits<float>::min();

    statistics[9]  = std::numeric_limits<float>::max();
    statistics[10] = std::numeric_limits<float>::max();
    statistics[11] = std::numeric_limits<float>::max();


    for (std::vector<int>::const_iterator iIt = neighbors.begin(); iIt != neighbors.end(); ++iIt)
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

    for (std::vector<int>::const_iterator iIt = neighbors.begin(); iIt != neighbors.end(); ++iIt)
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


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> bool
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::initCompute ()
{
    if (!Keypoint<PointInT, PointOutT>::initCompute ())
    {
        PCL_ERROR ("[pcl::%s::initCompute] init failed!\n", name_.c_str ());
        return (false);
    }

    if (method_ < 1 || method_ > 5)
    {
        PCL_ERROR ("[pcl::%s::initCompute] method (%d) must be in [1..5]!\n", name_.c_str (), method_);
        return (false);
    }

    if (!normals_)
    {
        PointCloudNPtr normals (new PointCloudN ());
        normals->reserve (normals->size ());
        if (!surface_->isOrganized ())
        {
            pcl::NormalEstimation<PointInT, NormalT> normal_estimation;
            normal_estimation.setInputCloud (surface_);
            normal_estimation.setRadiusSearch (search_radius_);
            normal_estimation.compute (*normals);
        }
        else
        {
            IntegralImageNormalEstimation<PointInT, NormalT> normal_estimation;
            normal_estimation.setNormalEstimationMethod (pcl::IntegralImageNormalEstimation<PointInT, NormalT>::SIMPLE_3D_GRADIENT);
            normal_estimation.setInputCloud (surface_);
            normal_estimation.setNormalSmoothingSize (5.0);
            normal_estimation.compute (*normals);
        }
        normals_ = normals;
    }
    if (normals_->size () != surface_->size ())
    {
        PCL_ERROR ("[pcl::%s::initCompute] normals given, but the number of normals does not match the number of input points!\n", name_.c_str (), method_);
        return (false);
    }
    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::detectKeypoints (PointCloudOut &output)
{
    boost::shared_ptr<pcl::PointCloud<PointOutT> > response (new pcl::PointCloud<PointOutT> ());

    response->points.reserve (input_->points.size());

    responseGP(*response);

    if (!nonmax_)
    {
        output = *response;
        // we do not change the denseness in this case
        output.is_dense = input_->is_dense;
    }
    else
    {
        output.points.clear ();
        output.points.reserve (response->points.size());

#ifdef _OPENMP
#pragma omp parallel for shared (output) num_threads(threads_)   
#endif
        for (int idx = 0; idx < static_cast<int> (response->points.size ()); ++idx)
        {
            if (!isFinite (response->points[idx]) ||
                    !pcl_isfinite (response->points[idx].intensity) ||
                    response->points[idx].intensity < threshold_)
                continue;

            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            tree_->radiusSearch (idx, search_radius_, nn_indices, nn_dists);
            bool is_maxima = true;
            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {
                if (response->points[idx].intensity < response->points[*iIt].intensity)
                {
                    is_maxima = false;
                    break;
                }
            }
            if (is_maxima)
#ifdef _OPENMP
#pragma omp critical
#endif
                output.points.push_back (response->points[idx]);
        }

        if (refine_)
            refineCorners (output);

        output.height = 1;
        output.width = static_cast<uint32_t> (output.points.size());
        output.is_dense = true;
    }
}

template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::setGeno(QString geno)
{
    _geno = QString(geno);
}

template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::computeKeypointsIndices(pcl::PointCloud<int> & indices)
{
    tree_->setInputCloud(input_);

    boost::shared_ptr<pcl::PointCloud<PointOutT> > response (new pcl::PointCloud<PointOutT> ());

    response->points.reserve (input_->points.size());


    responseGP(*response);
    //responseHarris(*response);
    if (!nonmax_)
    {
    }
    else
    {

        indices.clear();
        indices.reserve(response->points.size());
        for (int idx = 0; idx < static_cast<int> (response->points.size ()); ++idx)
        {
            if (!isFinite (response->points[idx]) ||
                    !pcl_isfinite (response->points[idx].intensity) ||
                    response->points[idx].intensity < threshold_)
                continue;
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;

            tree_->radiusSearch (idx, search_radius_, nn_indices, nn_dists);

            bool is_maxima = true;
            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {

                if (response->points[idx].intensity < response->points[*iIt].intensity)
                {
                    is_maxima = false;
                    break;
                }
            }
            if (is_maxima)
            {
                indices.push_back(idx);
            }
        }

        // no refine
    }
}

template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseGP(PointCloudOut &output)
{
    // populize domdocument
    QDomDocument doc;
    if (!doc.setContent(_geno))
    {
        qDebug() << "Fail to populize domdocument...";
        return;
    }

    cv::Mat_<float> result;

    CloudTPtr input_t(new CloudT);
    *input_t = *input_;
    TrainingData::extractFeatrues(input_t, _features, search_radius_);

    // detect keypoints for input cloud
    // get feature point detection result

    qDebug() << "Tag name: " << doc.firstChildElement().tagName();
    if(doc.firstChildElement().tagName() == ("GP"))
    {
        result = preOrder( doc.firstChildElement().firstChildElement() );
    }
    else
    {
        result = preOrder( doc.firstChildElement());
    }
    cv::Mat norm;

    // normalize to [0, 255]
    cv::normalize( result, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );

    output.resize(input_->size());

    cv::Size size = norm.size();
    if( norm.isContinuous())
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int i = 0; i < size.height; i++ )
    {

        uchar* p_img = (uchar*)(norm.data + norm.step*i);

        for( int j = 0; j < size.width; j++ )
        {
            int pIdx = j + i * size.width;
            const PointInT& pointIn = input_->points [pIdx];
            output [pIdx].intensity = 0.0;
            if (isFinite (pointIn))
            {
                output [pIdx].intensity =(float) p_img[j];
            }
            output [pIdx].x = pointIn.x;
            output [pIdx].y = pointIn.y;
            output [pIdx].z = pointIn.z;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseHarris (PointCloudOut &output) const
{
    PCL_ALIGN (16) float covar [8];
    float statistics[12];
    output.resize (input_->size ());
#ifdef _OPENMP
#pragma omp parallel for shared (output) private (covar) num_threads(threads_)
#endif

    // initialize feature matrix
//    int num_col = input_->size();
    //    int w = 459, h = 306;
    //    cv::Mat_<float> _xx(w, h), _xy(w, h), _xz(w, h), _yy(w, h), _yz(w, h), _zz(w, h);
    //    cv::Mat_<float> _xx(1, num_col), _xy(1, num_col), _xz(1, num_col), _yy(1, num_col), _yz(1, num_col), _zz(1, num_col);
    //    cv::Mat_<float> _avgx(1, num_col), _avgy(1, num_col), _avgz(1, num_col), _stdx(1, num_col), _stdy(1, num_col), _stdz(1, num_col);
    //    cv::Mat_<float> _maxx(1, num_col), _maxy(1, num_col), _maxz(1, num_col), _minx(1, num_col), _miny(1, num_col), _minz(1, num_col);

//    cv::Mat_<float> _f[18];
    // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
    // ax ay az sx sy sz bx by bz ix iy iz xx xy xz yy yz zz // avg std min max xx xy zz yy yz zz

//    for (int i = 0; i < 18; i++)
//    {
//        std::cout << "--initialize mat[" << i << "]: is continuous ? ";
//        _f[i] = cv::Mat_<float>(1, num_col);
//        if (_f[i].isContinuous()) std::cout << "yes!" << std::endl;
//        else std::cout << "no!" << std::endl;
//    }

    for (int pIdx = 0; pIdx < static_cast<int> (input_->size ()); ++pIdx)
    {
        const PointInT& pointIn = input_->points [pIdx];
        output [pIdx].intensity = 0.0; //std::numeric_limits<float>::quiet_NaN ();
        if (isFinite (pointIn))
        {
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;

            tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);
            //tree_->nearestKSearch(pointIn, 8, nn_indices, nn_dists);
            // Terminal Set1
            //            int normal_index = 0;
            //            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt, ++normal_index)
            //            {
            //                if (pcl_isfinite (normals_->points[*iIt].normal_x))
            //                {
            //                    ((float*)_f[3*normal_index + 0].data)[pIdx] = normals_->points[*iIt].normal_x;
            //                    ((float*)_f[3*normal_index + 1].data)[pIdx] = normals_->points[*iIt].normal_y;
            //                    ((float*)_f[3*normal_index + 2].data)[pIdx] = normals_->points[*iIt].normal_z;
            //                }
            //            }

            if (nn_indices.size() != 8) std::cout << "----No of Neighbors----" << nn_indices.size() << std::endl;
            //tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);
            calculateNormalCovar (nn_indices, covar);

            //            ((float*)_xx.data)[pIdx] = covar[0];
            //            ((float*)_xy.data)[pIdx] = covar[1];
            //            ((float*)_xz.data)[pIdx] = covar[2];

            //            ((float*)_yy.data)[pIdx] = covar[5];
            //            ((float*)_yz.data)[pIdx] = covar[6];
            //            ((float*)_zz.data)[pIdx] = covar[7];

            float trace = covar [0] + covar [5] + covar [7];
            if (trace != 0)
            {
                float det = covar [0] * covar [5] * covar [7] + 2.0f * covar [1] * covar [2] * covar [6]
                        - covar [2] * covar [2] * covar [5]
                        - covar [1] * covar [1] * covar [7]
                        - covar [6] * covar [6] * covar [0];

                output [pIdx].intensity = 0.04f + det - 0.04f * trace * trace;
            }
        }
        output [pIdx].x = pointIn.x;
        output [pIdx].y = pointIn.y;
        output [pIdx].z = pointIn.z;
    }

    //    cv::FileStorage fsxx("xx.xml", FileStorage::WRITE ); fsxx << "xx" << _xx; fsxx.release();
    //    cv::FileStorage fsxy("xy.xml", FileStorage::WRITE ); fsxy << "xy" << _xy; fsxy.release();
    //    cv::FileStorage fsxz("xz.xml", FileStorage::WRITE ); fsxz << "xz" << _xz; fsxz.release();
    //    cv::FileStorage fsyy("yy.xml", FileStorage::WRITE ); fsyy << "yy" << _yy; fsyy.release();
    //    cv::FileStorage fsyz("yz.xml", FileStorage::WRITE ); fsyz << "yz" << _yz; fsyz.release();
    //    cv::FileStorage fszz("zz.xml", FileStorage::WRITE ); fszz << "zz" << _zz; fszz.release();

    //    for (int i = 0; i < 24; i++)
    //    {
    //        QString filename("f_");
    //		filename += QString::number(i);
    //        cv::FileStorage file( (filename+".xml").toStdString(), FileStorage::WRITE );
    //		file << "f" << _f[i]; file.release();
    //		std::cout << filename.toStdString() << std::endl;
    //    }
    /*

  cv::imwrite("_xx.bmp", _xx); std::cout << _xx.cols << std::endl;
  cv::imwrite("_xy.bmp", _xy); std::cout << _xy.cols << std::endl;
  cv::imwrite("_xz.bmp", _xz); std::cout << _xz.cols << std::endl;

  cv::imwrite("_yy.bmp", _yy); std::cout << _yy.cols << std::endl;
  cv::imwrite("_yz.bmp", _yz); std::cout << _yz.cols << std::endl;
  cv::imwrite("_zz.bmp", _zz); std::cout << _zz.cols << std::endl;*/

    output.height = input_->height;
    output.width = input_->width;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseNoble (PointCloudOut &output) const
{
    PCL_ALIGN (16) float covar [8];
    output.resize (input_->size ());
#ifdef _OPENMP
#pragma omp parallel for shared (output) private (covar) num_threads(threads_)
#endif
    for (int pIdx = 0; pIdx < static_cast<int> (input_->size ()); ++pIdx)
    {
        const PointInT& pointIn = input_->points [pIdx];
        output [pIdx].intensity = 0.0;
        if (isFinite (pointIn))
        {
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);
            calculateNormalCovar (nn_indices, covar);
            // indices        0   1   2   3   4   5   6   7
            // coefficients: xx  xy  xz  ??  yx  yy  yz  ??
            float trace = covar [0] + covar [5] + covar [7];
            if (trace != 0)
            {
                float det = covar [0] * covar [5] * covar [7] + 2.0f * covar [1] * covar [2] * covar [6]
                        - covar [2] * covar [2] * covar [5]
                        - covar [1] * covar [1] * covar [7]
                        - covar [6] * covar [6] * covar [0];

                output [pIdx].intensity = det / trace;
            }
        }
        output [pIdx].x = pointIn.x;
        output [pIdx].y = pointIn.y;
        output [pIdx].z = pointIn.z;
    }
    output.height = input_->height;
    output.width = input_->width;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseLowe (PointCloudOut &output) const
{
    PCL_ALIGN (16) float covar [8];
    output.resize (input_->size ());
#ifdef _OPENMP
#pragma omp parallel for shared (output) private (covar) num_threads(threads_)
#endif
    for (int pIdx = 0; pIdx < static_cast<int> (input_->size ()); ++pIdx)
    {
        const PointInT& pointIn = input_->points [pIdx];
        output [pIdx].intensity = 0.0;
        if (isFinite (pointIn))
        {
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);
            calculateNormalCovar (nn_indices, covar);
            float trace = covar [0] + covar [5] + covar [7];
            if (trace != 0)
            {
                float det = covar [0] * covar [5] * covar [7] + 2.0f * covar [1] * covar [2] * covar [6]
                        - covar [2] * covar [2] * covar [5]
                        - covar [1] * covar [1] * covar [7]
                        - covar [6] * covar [6] * covar [0];

                output [pIdx].intensity = det / (trace * trace);
            }
        }
        output [pIdx].x = pointIn.x;
        output [pIdx].y = pointIn.y;
        output [pIdx].z = pointIn.z;
    }
    output.height = input_->height;
    output.width = input_->width;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseCurvature (PointCloudOut &output) const
{
    PointOutT point;
    for (unsigned idx = 0; idx < input_->points.size(); ++idx)
    {
        point.x = input_->points[idx].x;
        point.y = input_->points[idx].y;
        point.z = input_->points[idx].z;
        point.intensity = normals_->points[idx].curvature;
        output.points.push_back(point);
    }
    // does not change the order
    output.height = input_->height;
    output.width = input_->width;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::responseTomasi (PointCloudOut &output) const
{
    PCL_ALIGN (16) float covar [8];
    Eigen::Matrix3f covariance_matrix;
    output.resize (input_->size ());
#ifdef _OPENMP
#pragma omp parallel for shared (output) private (covar, covariance_matrix) num_threads(threads_)
#endif
    for (int pIdx = 0; pIdx < static_cast<int> (input_->size ()); ++pIdx)
    {
        const PointInT& pointIn = input_->points [pIdx];
        output [pIdx].intensity = 0.0;
        if (isFinite (pointIn))
        {
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);
            calculateNormalCovar (nn_indices, covar);

            float trace = covar [0] + covar [5] + covar [7];
            if (trace != 0)
            {
                covariance_matrix.coeffRef (0) = covar [0];
                covariance_matrix.coeffRef (1) = covariance_matrix.coeffRef (3) = covar [1];
                covariance_matrix.coeffRef (2) = covariance_matrix.coeffRef (6) = covar [2];
                covariance_matrix.coeffRef (4) = covar [5];
                covariance_matrix.coeffRef (5) = covariance_matrix.coeffRef (7) = covar [6];
                covariance_matrix.coeffRef (8) = covar [7];

                EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
                pcl::eigen33(covariance_matrix, eigen_values);
                output [pIdx].intensity = eigen_values[0];
            }
        }
        output [pIdx].x = pointIn.x;
        output [pIdx].y = pointIn.y;
        output [pIdx].z = pointIn.z;
    }
    output.height = input_->height;
    output.width = input_->width;
}
template <typename PointInT, typename PointOutT, typename NormalT> Mat_<float>
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::preOrder(QDomElement node)
{
    //******************GPManager XML******************
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
//template <typename PointInT, typename PointOutT, typename NormalT> void
//pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::extractFeatrues(const PointCloudIn &cloud, cv::Mat_<float> *features) const
//{
//    clock_t begin, end;
//    double time_spent;

//    begin = clock();
//    /* here, do your time-consuming job */

//    if (cloud->empty())
//    {
//        std::cout << "no cloud selected..." << std::endl;
//    }

//    // Calculate Normals
//    PointCloudNConstPtr normals_;

//    PointCloudNPtr normals (new PointCloudN ());
//    normals->reserve (normals->size ());
//    pcl::NormalEstimation<PointInT, NormalT> normal_estimation;
//    normal_estimation.setInputCloud (cloud);
//    normal_estimation.setKSearch(8); //***********
//    //normal_estimation.setRadiusSearch (search_radius_);
//    normal_estimation.compute (*normals);
//    normals_ = normals;

//    int num_col = cloud->size();
//    //cv::Mat_<float> _f[24];
//    for (int i = 0; i < 18; i++)
//    {
//        std::cout << "--initialize feature mat[" << i << "]: is continuous ? ";
//        features[i] = cv::Mat_<float>(1, num_col);
//        if (features[i].isContinuous()) std::cout << "yes!--" << std::endl;
//        else std::cout << "no!--" << std::endl;
//    }

//    pcl::KdTreeFLANN<PointInT>::Ptr tree_(new pcl::KdTreeFLANN<PointInT>());

//    tree_->setInputCloud (cloud);

//    for (int pIdx = 0; pIdx < static_cast<int> (cloud->size ()); ++pIdx)
//    {
//        const PointT& pointIn = cloud->points [pIdx];

//        if (isFinite (pointIn))
//        {
//            std::vector<int> nn_indices;
//            std::vector<float> nn_dists;

//            // n-nearest number search
//            tree_->nearestKSearch(pointIn, 8, nn_indices, nn_dists);
//            // radius search
//            //float search_radius_ = 0.15;
//            //tree_->radiusSearch (pointIn, search_radius_, nn_indices, nn_dists);

//            // Terminal set2
//            unsigned count = 0;
//            float statistics[12];
//            // calculate 0 ~ 11
//            // avg 0 1 2; std 3 4 5; max 6 7 8; min 9 10 11;
//            memset (statistics, 0, sizeof (float) * 12);

//            statistics[6]  = std::numeric_limits<float>::min();
//            statistics[7]  = std::numeric_limits<float>::min();
//            statistics[8]  = std::numeric_limits<float>::min();

//            statistics[9]  = std::numeric_limits<float>::max();
//            statistics[10] = std::numeric_limits<float>::max();
//            statistics[11] = std::numeric_limits<float>::max();


//            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
//            {
//                if (pcl_isfinite (normals_->points[*iIt].normal_x))
//                {
//                    statistics[0] += normals_->points[*iIt].normal_x;
//                    statistics[1] += normals_->points[*iIt].normal_y;
//                    statistics[2] += normals_->points[*iIt].normal_z;

//                    if ( statistics[6]  < normals_->points[*iIt].normal_x ) statistics[6]  = normals_->points[*iIt].normal_x;
//                    if ( statistics[7]  < normals_->points[*iIt].normal_y ) statistics[7]  = normals_->points[*iIt].normal_y;
//                    if ( statistics[8]  < normals_->points[*iIt].normal_z ) statistics[8]  = normals_->points[*iIt].normal_z;

//                    if ( statistics[9]  > normals_->points[*iIt].normal_x ) statistics[9]  = normals_->points[*iIt].normal_x;
//                    if ( statistics[10] > normals_->points[*iIt].normal_y ) statistics[10] = normals_->points[*iIt].normal_y;
//                    if ( statistics[11] > normals_->points[*iIt].normal_z ) statistics[11] = normals_->points[*iIt].normal_z;

//                    ++count;
//                }
//            }

//            float norm;
//            if (count > 0)
//            {
//                norm = 1.0 / float (count);
//                statistics[0] *= norm;
//                statistics[1] *= norm;
//                statistics[2] *= norm;
//            }

//            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
//            {
//                if (pcl_isfinite (normals_->points[*iIt].normal_x))
//                {
//                    statistics[3] += sqrt((normals_->points[*iIt].normal_x - statistics[0])*(normals_->points[*iIt].normal_x - statistics[0]));
//                    statistics[4] += sqrt((normals_->points[*iIt].normal_y - statistics[1])*(normals_->points[*iIt].normal_y - statistics[1]));
//                    statistics[5] += sqrt((normals_->points[*iIt].normal_z - statistics[2])*(normals_->points[*iIt].normal_z - statistics[2]));
//                }
//            }

//            if (count > 0)
//            {
//                norm = 1.0 / float (count);
//                statistics[3] *= norm;
//                statistics[4] *= norm;
//                statistics[5] *= norm;
//            }

//            // save 0 ~ 11
//            for (int i = 0; i < 12; i++)
//            {
//                ((float*)features[i].data)[pIdx] = statistics[i];
//            }

//            // calculate 12 ~ 17
//            float coefficients[6];
//            memset (coefficients, 0, sizeof (float) * 6);
//            count = 0;
//            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
//            {
//                if (pcl_isfinite (normals_->points[*iIt].normal_x))
//                {
//                    coefficients[0] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_x;
//                    coefficients[1] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_y;
//                    coefficients[2] += normals_->points[*iIt].normal_x * normals_->points[*iIt].normal_z;

//                    coefficients[3] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_y;
//                    coefficients[4] += normals_->points[*iIt].normal_y * normals_->points[*iIt].normal_z;
//                    coefficients[5] += normals_->points[*iIt].normal_z * normals_->points[*iIt].normal_z;

//                    ++count;
//                }
//            }

//            if (count > 0)
//            {
//                float norm = 1.0 / float (count);
//                coefficients[0] *= norm;
//                coefficients[1] *= norm;
//                coefficients[2] *= norm;
//                coefficients[3] *= norm;
//                coefficients[4] *= norm;
//                coefficients[5] *= norm;
//            }
//            // save 12 ~ 17
//            for (int i = 0; i < 6; i++)
//            {
//                ((float*)features[i+12].data)[pIdx] = coefficients[i];
//            }

//            // Terminal set1
//            //            int normal_index = 0;
//            //            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt, ++normal_index)
//            //            {
//            //                if (pcl_isfinite (normals_->points[*iIt].normal_x))
//            //                {
//            //                    ((float*)features[3*normal_index + 0].data)[pIdx] = normals_->points[*iIt].normal_x;
//            //                    ((float*)features[3*normal_index + 1].data)[pIdx] = normals_->points[*iIt].normal_y;
//            //                    ((float*)features[3*normal_index + 2].data)[pIdx] = normals_->points[*iIt].normal_z;
//            //                }
//            //            }
//        }
//    }

//    // output calculating time
//    std::cout << "Feature Extracted...." << std::endl;
//    end = clock();
//    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//    std::cout << "Processing Time: " << time_spent << "s;" << std::endl;
//}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointInT, typename PointOutT, typename NormalT> void
pcl::GPKeypoint3D<PointInT, PointOutT, NormalT>::refineCorners (PointCloudOut &corners) const
{
    Eigen::Matrix3f nnT;
    Eigen::Matrix3f NNT;
    Eigen::Matrix3f NNTInv;
    Eigen::Vector3f NNTp;
    float diff;
    const unsigned max_iterations = 10;
#ifdef _OPENMP
#pragma omp parallel for shared (corners) private (nnT, NNT, NNTInv, NNTp, diff) num_threads(threads_)
#endif
    for (int cIdx = 0; cIdx < static_cast<int> (corners.size ()); ++cIdx)
    {
        unsigned iterations = 0;
        do {
            NNT.setZero();
            NNTp.setZero();
            PointInT corner;
            corner.x = corners[cIdx].x;
            corner.y = corners[cIdx].y;
            corner.z = corners[cIdx].z;
            std::vector<int> nn_indices;
            std::vector<float> nn_dists;
            tree_->radiusSearch (corner, search_radius_, nn_indices, nn_dists);
            for (std::vector<int>::const_iterator iIt = nn_indices.begin(); iIt != nn_indices.end(); ++iIt)
            {
                if (!pcl_isfinite (normals_->points[*iIt].normal_x))
                    continue;

                nnT = normals_->points[*iIt].getNormalVector3fMap () * normals_->points[*iIt].getNormalVector3fMap ().transpose();
                NNT += nnT;
                NNTp += nnT * surface_->points[*iIt].getVector3fMap ();
            }
            if (invert3x3SymMatrix (NNT, NNTInv) != 0)
                corners[cIdx].getVector3fMap () = NNTInv * NNTp;

            diff = (corners[cIdx].getVector3fMap () - corner.getVector3fMap()).squaredNorm ();
        } while (diff > 1e-6 && ++iterations < max_iterations);
    }
}

#define PCL_INSTANTIATE_HarrisKeypoint3D(T,U,N) template class PCL_EXPORTS pcl::HarrisKeypoint3D<T,U,N>;
#endif // #ifndef PCL_HARRIS_KEYPOINT_3D_IMPL_H_

