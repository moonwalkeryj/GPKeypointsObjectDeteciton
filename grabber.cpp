#include "grabber.h"
#include <QTimer>
#include <QtConcurrentRun>
#include "controller.h"
Grabber::Grabber():
    grabber_()
{

}

Grabber::Grabber(QObject *controller)
{
    controller_ = (Controller*)controller;
}

Grabber::~Grabber()
{
    closeGrabber();
}

void Grabber::startGrabber()
{
    QtConcurrent::run (boost::bind (&Grabber::startGrabberImpl, this));
}

void Grabber::closeGrabber()
{
    if (grabber_ && grabber_->isRunning ()) grabber_->stop ();
    if (new_data_connection_.connected ())  new_data_connection_.disconnect ();
}

void Grabber::startGrabberImpl ()
{
    starting_grabber_ = true;

    try
    {
        grabber_ = CPGrabberPtr (new CPGrabber ());
    }
    catch (const pcl::PCLException& e)
    {
        std::cerr << "ERROR in in_hand_scanner.cpp: " << e.what () << std::endl;
        exit (EXIT_FAILURE);
    }

    boost::function <void (const CloudTConstPtr&)> new_data_cb = boost::bind (&Controller::newDataCallbackXYZRGBA, controller_, _1);
    new_data_connection_ = grabber_->registerCallback (new_data_cb);

    grabber_->start ();
    starting_grabber_ = false;
}
