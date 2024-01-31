//
// Created by luigi on 31/01/24.
//

#ifndef OMPKMEANS_POINT_H
#define OMPKMEANS_POINT_H

#define DIMENSION 2
struct Point{
    double coords[DIMENSION];
    int clusterLabel;
    Point(double* coordinates){
        for(int i = 0;i< DIMENSION; ++i){
            coords[i] = coordinates[i];
        }
        this->clusterLabel = -1;
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
