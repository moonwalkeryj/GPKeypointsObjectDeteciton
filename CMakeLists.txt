cmake_minimum_required(VERSION 2.8 FATAL_ERROR) 
project(pclkeypoint)
set(EXE_NAME pcl_keypoint_v2)
set(PROJECT_NAME pcl_keypoint)
set(DEFAULT FALSE) # I dont know the usage
set(ADDITIONAL_LIBS "")

# Package for OpenCV
include(C:/OpenCV/build/install/OpenCVConfig.cmake)
set(OpenCV "C:/OpenCV/build/cmake")
find_package(OpenCV REQUIRED)

# Package for PCL
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
list(APPEND ADDITIONAL_LIBS ${PCL_LIBRARIES})

# Package for VTK (is it necessary to use this, since it can be included by PCL)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

# Package for Qt
find_package(Qt4 REQUIRED)
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL QtXml REQUIRED)
include(${QT_USE_FILE})
list(APPEND ADDITIONAL_LIBS ${QT_LIBRARIES})

################################################################################
set(SUBSYS_NAME pcl_processor)
set(SUBSYS_DESC "My PCL Processor")
set(SUBSYS_DEPS     common     features     io     kdtree apps)
set(SUBSYS_LIBS)
################################################################################

if(BUILD)
  # Qt
  if(NOT QT4_FOUND)
    message(WARNING "Qt4 is needed for the app! It will not be built!")
    set(BUILD FALSE)
  endif()

  # OpenGL
  find_package(OpenGL REQUIRED)
  if(OPENGL_FOUND AND OPENGL_GLU_FOUND)
    list(APPEND ADDITIONAL_LIBS ${OPENGL_LIBRARIES})
  else()
    message(WARNING "OpenGL & GLU are needed for the app! It will not be built!")
    set(BUILD FALSE)
  endif()
endif()
################################################################################

set(INCS
        cloudprocessor.h
        harris_3d.h
)

set(IMPL_INCS
        cloudprocessor.hpp
        harris_3d.hpp
)

set(SRCS
        vtkwidget.cpp
        harris_3d.cpp
        qpointcloudwidget.cpp
        main.cpp
        mainwindow.cpp
        cloudprocessor.cpp
        controller.cpp
        trainingdata.cpp
        objectdetector.cpp
)

# Qt Moc Files ( Classes using Macro Q_OBJECT)
set(MOC_VTK_WIDGET_INC                  vtkwidget.h)
set(MOC_CONTROLLER_INC                  controller.h)
set(MOC_QPOINTCLOUD_WIDGET_INC          qpointcloudwidget.h)
set(MOC_MAIN_WINDOW_INC 		mainwindow.h)
set(MOC_TRAINING_DATA_INC 		trainingdata.h)
set(MOC_OBJECT_DETECTOR_INC 		objectdetector.h)
set(UI_MAIN_WINDOW                      mainwindow.ui)

QT4_WRAP_CPP(MOC_VTK_WIDGET_SRC         ${MOC_VTK_WIDGET_INC})
QT4_WRAP_CPP(MOC_QPOINTCLOUD_WIDGET_SRC ${MOC_QPOINTCLOUD_WIDGET_INC})
QT4_WRAP_CPP(MOC_CONTROLLER_SRC         ${MOC_CONTROLLER_INC})
QT4_WRAP_CPP(MOC_MAIN_WINDOW_SRC        ${MOC_MAIN_WINDOW_INC})
QT4_WRAP_CPP(MOC_TRAINING_DATA_SRC 	${MOC_TRAINING_DATA_INC})
QT4_WRAP_CPP(MOC_OBJECT_DETECTOR_SRC 	${MOC_OBJECT_DETECTOR_INC})
QT4_WRAP_UI(UI_MAIN_WINDOW_INC          ${UI_MAIN_WINDOW})

include_directories(${CMAKE_CURRENT_BINARY_DIR}) # For the ui files


list(APPEND INCS    ${MOC_VTK_WIDGET_INC}
                    ${MOC_QPOINTCLOUD_WIDGET_INC}
                    ${MOC_CONTROLLER_INC}
                    ${MOC_MAIN_WINDOW_INC}
                    ${MOC_TRAINING_DATA_INC}
                    ${MOC_OBJECT_DETECTOR_INC}
                    ${UI_MAIN_WINDOW_INC}

)

list(APPEND SRCS    ${MOC_VTK_WIDGET_SRC}
                    ${MOC_QPOINTCLOUD_WIDGET_SRC}
                    ${MOC_CONTROLLER_SRC}
                    ${MOC_MAIN_WINDOW_SRC}
                    ${MOC_TRAINING_DATA_SRC}
                    ${MOC_OBJECT_DETECTOR_SRC}
)

add_executable(${EXE_NAME} ${SRCS} ${INCS} ${IMPL_INCS})
target_link_libraries(${EXE_NAME} ${SUBSYS_LIBS} ${ADDITIONAL_LIBS} ${OpenCV_LIBS} GlU32 QVTK)

