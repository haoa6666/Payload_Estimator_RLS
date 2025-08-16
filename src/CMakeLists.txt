cmake_minimum_required(VERSION 3.0.2)
project(payload_estimator)

## 设置编译选项
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_CXX_STANDARD 17)

## 查找 catkin 的构建系统和依赖项
find_package(catkin REQUIRED COMPONENTS 
  roscpp
  roslib
  std_msgs
  message_generation
  urdf
  kdl_parser
)

find_package(Eigen3 REQUIRED)

## 生成消息
add_message_files(
  FILES
  ArmJointState.msg
  Debug.msg
  PayloadDebug.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  
)

## 声明 catkin 包
catkin_package(
  CATKIN_DEPENDS roscpp 
)

## 包含头文件路径
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/lib/
  ${PROJECT_SOURCE_DIR}
)

## 添加可执行文件
add_executable(payload_estimator
  payload_estimator/payload_estimator.cpp
  main.cpp
)

## 链接依赖库
target_link_libraries(payload_estimator
  ${catkin_LIBRARIES}
  glog
  gflags
)

add_dependencies(payload_estimator ${PROJECT_NAME}_generate_messages_cpp)
