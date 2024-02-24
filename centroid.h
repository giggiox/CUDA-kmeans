//
// Created by luigi on 18/02/24.
//

#ifndef OMPKMEANS_CENTROID_H
#define OMPKMEANS_CENTROID_H

#include <sstream>

struct Centroid{
    double x,y,z;
    int cardinality;

    Centroid(): x(0),y(0),z(0),cardinality(0){}

    Centroid(double nx, double ny, double nz): cardinality(0){
        x = nx; y = ny; z = nz;
    }

    std::string toString(){
        std::stringstream ss;
        ss << x << "," << y << "," << z << "," << cardinality;
        return ss.str();
    }

};

#endif //OMPKMEANS_CENTROID_H