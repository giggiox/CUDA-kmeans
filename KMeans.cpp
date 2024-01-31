//
// Created by luigi on 31/01/24.
//

#include <vector>
#include <cfloat>
#include "iostream"
#include "KMeans.h"
KMeans::KMeans(int k) {
    this->k = k;
}



std::vector<Point> KMeans::fit(std::vector<Point>& dataPoints, int maxIteration) {
    //Point p1 = Point();
    //p1.coords = {2.0,2.0};
    //p1.clusterLabel = -1;
    std::vector<Point> centroids = {};

    
    return centroids;
}


double KMeans::euclideanNorm(Point &p1, Point &p2) {
    return 0.0;
}

