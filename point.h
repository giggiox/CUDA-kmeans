//
// Created by luigi on 31/01/24.
//

#ifndef OMPKMEANS_POINT_H
#define OMPKMEANS_POINT_H
#include <sstream>

struct Point{
    float x,y,z;
    int clusterLabel;

    Point(): x(0),y(0),z(0),clusterLabel(-1){}

    Point(float nx, float ny, float nz): clusterLabel(-1){
        x = nx; y = ny; z = nz;
    }

    std::string toString(){
        std::stringstream ss;
        ss << x << "," << y << "," << z << "," << clusterLabel;
        return ss.str();
    }

};

#endif //OMPKMEANS_POINT_H