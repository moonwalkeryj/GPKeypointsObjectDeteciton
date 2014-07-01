#include "qpointcloudwidget.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/common/common.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <pcl/io/pcd_io.h>
#include <pcl/exceptions.h>

#include <math.h>
#include <QtCore>

#include <opencv2\opencv.hpp>

QPointCloudWidget::QPointCloudWidget (QWidget *parent)
    :QVTKWidget(parent),
      mutex_vis_             (),
      cloud(new CloudT),
      pviz(new pcl::visualization::PCLVisualizer("", false)),
      timer(new QTimer(this))
{
    memset(&cam_PosParas[0], 0, 9*sizeof(double));
    cam_PosParas[POSZ] = -10.0; cam_PosParas[VIEWZ] = -1.0; cam_PosParas[UPY] = -1.0;

    /*
	cv::Mat _desired_cormsk = cv::Mat::zeros(459, 306, CV_8UC1);
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
*/
    // this creates and displays a window named "test_viz"
    // upon calling PCLVisualizerInteractor interactor_->Initialize ();
    // how to disable that?
    //        pcl::visualization::PCLVisualizer pviz ("test_viz");
    //        pcl::visualization::PCLVisualizer pviz2("ok");
	pviz->setBackgroundColor(0, 0, 0);
    pviz->addCoordinateSystem(1.0f, 0.0, 0.0, 0.0);
    pviz->initCameraParameters();

    // Initialize window
    this->SetRenderWindow(pviz->getRenderWindow());
    pviz->setupInteractor(this->GetInteractor(), this->GetRenderWindow());
    pviz->getInteractorStyle ()->setKeyboardModifier (pcl::visualization::INTERACTOR_KB_MOD_SHIFT);
    this->viewupdate();
    /*
    pviz->addPointCloud<PointT>(cloud_xyz, "cloud1");*/

}

QPointCloudWidget::~QPointCloudWidget()
{
    this->SetRenderWindow(NULL);
    pviz->removeAllPointClouds();
    delete pviz;
}

void
QPointCloudWidget::resetCamera()
{
    //pviz->setCameraPosition(cam_PosParas[0], cam_PosParas[1], cam_PosParas[2], cam_PosParas[3], cam_PosParas[4], cam_PosParas[5],cam_PosParas[6], cam_PosParas[7], cam_PosParas[8]);
    pviz->setCameraPosition(cam_PosParas[POSX],cam_PosParas[POSY],cam_PosParas[POSZ],cam_PosParas[UPX],cam_PosParas[UPY],cam_PosParas[UPZ],0);
    //pviz->resetCameraViewpoint();
    viewupdate();
}

void
QPointCloudWidget::setCameraParameters(int pos, double value)
{
    cam_PosParas[pos] = value;
    resetCamera();
}

void
QPointCloudWidget::mousePressEvent (QMouseEvent *event){
    cerr << "---VTKWidget::mousePressEvent-----" << endl;
    mutex_vis_.lock();
    QVTKWidget::mousePressEvent(event);
}
void
QPointCloudWidget::mouseMoveEvent (QMouseEvent *event){
    QVTKWidget::mouseMoveEvent(event);
}
void
QPointCloudWidget::mouseReleaseEvent (QMouseEvent *event){
    cerr << "---VTKWidget::mouseReleaseEvent-----" << endl;
    mutex_vis_.unlock();
    QVTKWidget::mouseReleaseEvent(event);
}
void
QPointCloudWidget::wheelEvent (QWheelEvent *event){
    QVTKWidget::wheelEvent(event);
}

void QPointCloudWidget::paintEvent(QPaintEvent *event)
{
    //boost::mutex::scoped_lock lock (mutex_vis_);
    QVTKWidget::paintEvent(event);
}

void QPointCloudWidget::resizeEvent(QResizeEvent *event)
{
    QVTKWidget::resizeEvent(event);
}

void QPointCloudWidget::viewupdate()
{
    update();
}

pcl::visualization::PCLVisualizer *QPointCloudWidget::getPCLVisualizer()
{
    return pviz;
}

void QPointCloudWidget::showPointCloud(const CloudTConstPtr &cloud)
{
    static bool count = true;
    // update view
    if (count) {
        pviz->removeAllPointClouds();
        pviz->addPointCloud(cloud, "cloud");
        count = false;
    }
    pviz->updatePointCloud(cloud, "cloud");
    resetCamera();
    update();
}

/*
void QPointCloudWidget::showPointCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud)
{
    static bool count = true;
    // update view
    if (count) {
        pviz->removeAllPointClouds();
        pviz->addPointCloud(cloud);
        count = false;
    }
    pviz->updatePointCloud(cloud);
    update();
}
*/
