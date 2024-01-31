cmake_minimum_required(VERSION 3.27)
project(ompkmeans)

set(CMAKE_CXX_STANDARD 17)

add_executable(ompkmeans main.cpp
        KMeans.cpp
        KMeans.h
        point.h)
