#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <pcl/io/openni_grabber.h>
#include <boost/thread/mutex.hpp>
#include <QImage>
#include <QTimer>

#include "controller.h"

namespace Ui {
class MainWindow;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    //typedef boost::shared_ptr<Controller> ControllerPtr;

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_actionOpen_PCL_triggered();
    void on_actionOpenKeypoints_triggered(bool checked);
    void on_actionExtract_Features_triggered();

    void on_actionGP_Keypoints_triggered(bool checked);

    void on_actionRotateX5Degrees_triggered();

    void on_pushButton_Rotate_clicked();
    void on_pushButton_Translate_clicked();

    void on_actionSave_Cloud_triggered();

    void on_actionSet_Training_Target_triggered();

    void on_actionSave_Training_Features_triggered();

    void on_actionDetection_Set_Scene_triggered();

    void on_actionDetection_Set_Model_triggered();

signals:
    void GPKeypointsShow(bool checked, QString s);
    void rotate(int degrees, int axis);
    void translate(int distance, int axis);
    void savePointCloud(QString filename);
    void saveKeypoints(QString filename);

    void openTrainingTarget(QString filename);
    void saveTrainingFeaturesAndTarget(QString filename);



private:
    Controller* _controller;
    Ui::MainWindow *ui;

};

#endif // MAINWINDOW_H
