#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/common.h>
#include <pcl/exceptions.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/geometry/get_boundary.h>
#include <pcl/geometry/mesh_conversion.h>
#include <QApplication>
#include <QtCore>
#include <QFileDialog>
#include <opencv2/opencv.hpp>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    QMainWindow::showMaximized();
    _controller = new Controller(ui->pointcloudwidget);
    QObject::connect(ui->actionResetCamera, SIGNAL(triggered()), _controller, SLOT(resetCameraView()) );
    QObject::connect(ui->horizontalSlider, SIGNAL(sliderMoved(int)), _controller, SLOT(setGpThreshold(int)) );

    QObject::connect(ui->actionHarris3D_Keypoints, SIGNAL(triggered(bool)), _controller, SLOT(toggleShowHarrisKeypoints(bool)) );
    QObject::connect(ui->actionISS_Keypoints, SIGNAL(triggered(bool)), _controller, SLOT(toggleShowISSKeypoints(bool)) );

    QObject::connect(ui->actionRotateX5Degrees, SIGNAL(triggered()), _controller, SLOT(rotateX()) );

    QObject::connect(this, SIGNAL(GPKeypointsShow(bool,QString)), _controller, SLOT(toggleShowGPKeypoints(bool,QString)) );
    QObject::connect(this, SIGNAL(rotate(int,int)), _controller, SLOT(rotate(int,int)) );
    QObject::connect(this, SIGNAL(translate(int,int)), _controller, SLOT(translate(int,int)));
    QObject::connect(this, SIGNAL(savePointCloud(QString)), _controller, SLOT(savePointCloud(QString)));

    QObject::connect(this, SIGNAL(openTrainingTarget(QString)), &(_controller->_trainingData), SLOT(setTarget(QString)));
    QObject::connect(ui->actionAppend_Training_Features, SIGNAL(triggered()), _controller, SLOT(appendTrainingFeatures()));
    QObject::connect(this, SIGNAL(saveTrainingFeaturesAndTarget(QString)), &(_controller->_trainingData), SLOT(saveTrainingFeaturesAndTarget(QString)));

    QObject::connect(ui->actionDetection_Detect, SIGNAL(triggered()), _controller, SLOT(objectDetectionTest())) ;

}

MainWindow::~MainWindow()
{
    delete _controller;
    delete ui;
}

void MainWindow::on_actionOpen_PCL_triggered()
{
    QString filename =
            QFileDialog::getOpenFileName(this,
                                         tr("Open Point Cloud"), ".",
                                         tr("Point Cloud (*.pcd *.ply)"));
    if(filename.isEmpty()) return;
	std::cout<<filename.toStdString();
    _controller->showPointCloud( filename.toStdString());
	
}

void MainWindow::on_actionExtract_Features_triggered()
{

}

void MainWindow::on_actionOpenKeypoints_triggered(bool checked)
{
    if (checked)
    {
        QString filename =
                QFileDialog::getOpenFileName(this,
                                             tr("Open Keypoints File"), ".",
                                             tr("Key Points (*.pcd *.ply *.xml)"));
        if(filename.isEmpty()) return;
        std::cout << filename.toStdString();
        _controller->showKeypoints( filename.toStdString());
    }
    else
    {
        _controller->removeKeypoints();
    }
}


void MainWindow::on_actionGP_Keypoints_triggered(bool checked)
{
    emit GPKeypointsShow(checked, ui->textEdit->toPlainText());
}

void MainWindow::on_actionRotateX5Degrees_triggered()
{

}

void MainWindow::on_pushButton_Rotate_clicked()
{
    emit rotate( ui->spinBox_RotateDegree->value(), ui->comboBox_RotateAxis->currentIndex());
}

void MainWindow::on_pushButton_Translate_clicked()
{
    emit translate( ui->spinBox_TranslateDistance->value(), ui->comboBox_TranslateAxis->currentIndex());
}

void MainWindow::on_actionSave_Cloud_triggered()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Keypoints File"), ".",
                                                    tr("Key Points (*.pcd *.ply)"));
    if(filename.isEmpty()) return;
    std::cout << "Going to save " << filename.toStdString() << std::endl;
    emit savePointCloud(filename);
    emit saveKeypoints(QString("KE_") + filename);
    //_controller->showKeyPoints( filename.toStdString());
}

void MainWindow::on_actionSet_Training_Target_triggered()
{
	QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("Set Training Target File"), ".",
                                                    tr("Key Points (*.*)"));
    if(filename.isEmpty()) return;
    std::cout << "Training target file: " << filename.toStdString() << std::endl;
    emit openTrainingTarget(filename);
}

void MainWindow::on_actionSave_Training_Features_triggered()
{
    QString filename = QFileDialog::getSaveFileName(this,
                                                    tr("Save Keypoints File"), ".",
                                                    tr("Key Points (*.*)"));
    if(filename.isEmpty()) return;
    std::cout << "Going to save " << filename.toStdString() << std::endl;
    emit saveTrainingFeaturesAndTarget(filename);
}

void MainWindow::on_actionDetection_Set_Scene_triggered()
{
    QString filename =
            QFileDialog::getOpenFileName(this,
                                         tr("Set Scene Point Cloud"), ".",
                                         tr("Point Cloud (*.pcd *.ply)"));
    if(filename.isEmpty()) return;
    std::cout<<filename.toStdString();
    _controller->setScene(filename.toStdString());
}

void MainWindow::on_actionDetection_Set_Model_triggered()
{
    QString filename =
            QFileDialog::getOpenFileName(this,
                                         tr("Set Model Point Cloud"), ".",
                                         tr("Point Cloud (*.pcd *.ply)"));
    if(filename.isEmpty()) return;
    std::cout<<filename.toStdString();
    _controller->setModel(filename.toStdString());
}
