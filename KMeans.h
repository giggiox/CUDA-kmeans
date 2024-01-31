//
// Created by luigi on 31/01/24.
//

#ifndef OMPKMEANS_KMEANS_H
#define OMPKMEANS_KMEANS_H


#include "point.h"
#include <vector>

class KMeans {
private:
    int k;
    std::vector<Point> centroids;
public:
    KMeans(int k);
    std::vector<Point> fit(std::vector<Point>& dataPoints,int maxIteration);
    double euclideanNorm(Point& p1,Point& p2);
};


#endif //OMPKMEANS_KMEANS_H
