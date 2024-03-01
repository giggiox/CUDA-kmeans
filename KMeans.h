//
// Created by luigi on 31/01/24.
//

#ifndef OMPKMEANS_KMEANS_H
#define OMPKMEANS_KMEANS_H


#include "point.h"
#include "centroid.h"
#include <vector>

class KMeans {
private:
    int k;

public:
    std::vector<Centroid> centroids;
    KMeans(int k);
    void fit(std::vector<Point>& dataPoints,int maxIteration, bool useStopCondition);
    void fitParallel(std::vector<Point>& dataPoints, int maxIteration, bool useStopCondition);
    static float euclideanNorm(Point& p1,Centroid& p2);
    void assignRandomCentroids(std::vector<Point> dataPoints);
};


#endif //OMPKMEANS_KMEANS_H