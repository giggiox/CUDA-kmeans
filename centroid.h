//
// Created by luigi on 18/02/24.
//

#ifndef OMPKMEANS_CENTROID_H
#define OMPKMEANS_CENTROID_H
#include <array>

#define DIMENSION 3
struct Centroid{
    std::array<double, DIMENSION> coords;
    int cardinality;

    Centroid(): cardinality(0){
        for(int i = 0;i< DIMENSION; ++i){
            coords[i] = 0.0;
        }
    }

    Centroid(double* coordinates): cardinality(0){
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
        return out;
    }

};

#endif //OMPKMEANS_CENTROID_H
