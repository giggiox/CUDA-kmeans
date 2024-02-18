//
// Created by luigi on 31/01/24.
//

#ifndef OMPKMEANS_POINT_H
#define OMPKMEANS_POINT_H

#include <array>

#define DIMENSION 3
struct Point{
    std::array<double, DIMENSION> coords;
    int clusterLabel;

    Point(): clusterLabel(-1){
        for(int i = 0;i< DIMENSION; ++i){
            coords[i] = 0.0;
        }
    }

    Point(double* coordinates): clusterLabel(-1){
        for(int i = 0;i< DIMENSION; ++i){
            coords[i] = coordinates[i];
        }
    }

    std::string toString(){
        std::string out = "";
        for(int i = 0;i < DIMENSION; ++i){
                out += std::to_string(coords[i]);
                out += ",";
        }
        out += std::to_string(clusterLabel);
        return out;
    }

};

#endif //OMPKMEANS_POINT_H
