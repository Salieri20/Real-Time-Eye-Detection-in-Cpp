cmake_minimum_required(VERSION 3.20)
project(Movenet_working CXX)

find_package(tensorflowlite REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor REQUIRED)


add_executable(Movenet_working tflite_cpp.cpp) 

target_link_libraries(Movenet_working PRIVATE xtensor tensorflow::tensorflowlite opencv::opencv) 
