//
// Created by luigi on 31/01/24.
//

#include <vector>
#include <cfloat>
#include "iostream"
#include "KMeans.h"
#include "centroid.h"
#include <cmath>
#include <random>

KMeans::KMeans(int k) {
    this->k = k;
}

float KMeans::euclideanNorm(Point &p1, Centroid &p2) {
    return (p1.x-p2.x) * (p1.x-p2.x) + (p1.y-p2.y) * (p1.y-p2.y) + (p1.z-p2.z) * (p1.z-p2.z);
}


void KMeans::assignRandomCentroids(std::vector<Point> dataPoints){
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(0,dataPoints.size()-1);
    std::vector<Centroid> newCentroids;
    for(int i = 0;i<this->k;++i){
        int index = dist6(rng);
        Point p = dataPoints[index];
        newCentroids.push_back(Centroid(p.x,p.y,p.z));
    }
    this->centroids = newCentroids;
}




void KMeans::fit(std::vector<Point>& dataPoints, int maxIteration, bool useStopCondition) {
    if(this->centroids.empty()){
        std::cerr << "assign centroids first" << std::endl;
        return;
    }

    for (int _ = 0; _ < maxIteration;++_) {
        std::vector<Centroid> newCentroids = std::vector<Centroid>(this->k, Centroid());
        for (Point& p: dataPoints) {
            float minDistance = INFINITY;
            for (int j = 0; j < this->k; ++j) {
                float distance = euclideanNorm(p, this->centroids[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    p.clusterLabel = j;
                }
            }
            newCentroids[p.clusterLabel].x += p.x;
            newCentroids[p.clusterLabel].y += p.y;
            newCentroids[p.clusterLabel].z += p.z;
            newCentroids[p.clusterLabel].cardinality += 1;
        }
        bool converged = true;
        for (int i = 0; i < this->k; i++) {
            newCentroids[i].x /= newCentroids[i].cardinality;
            newCentroids[i].y /= newCentroids[i].cardinality;
            newCentroids[i].z /= newCentroids[i].cardinality;
            if (useStopCondition && std::abs(centroids[i].x - newCentroids[i].x) > 1e-15
                && std::abs(centroids[i].y - newCentroids[i].y) > 1e-15
                && std::abs(centroids[i].z - newCentroids[i].z) > 1e-15) {
                converged = false;
            }
        }

        if(useStopCondition && converged) {
            //std::cout << "stopped at iter: ";
            //std::cout << _ << std::endl;
            return;
        }

        this->centroids = newCentroids;
    }
}


void KMeans::fitParallel(std::vector<Point>& dataPoints, int maxIteration, bool useStopCondition) {
    if(this->centroids.empty()){
        std::cerr << "assign centroids first" << std::endl;
        return;
    }

    for (int _ = 0; _ < maxIteration;++_) {
        std::vector<Centroid> newCentroids = std::vector<Centroid>(this->k, Centroid());
#pragma omp parallel for schedule(static) default(none) shared(dataPoints,newCentroids) num_threads(4)
        for (Point& p: dataPoints) {
            float minDistance = INFINITY;
            for (int j = 0; j < this->k; ++j) {
                float distance = euclideanNorm(p, this->centroids[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    p.clusterLabel = j;
                }
            }
            newCentroids[p.clusterLabel].x += p.x;
            newCentroids[p.clusterLabel].y += p.y;
            newCentroids[p.clusterLabel].z += p.z;
            newCentroids[p.clusterLabel].cardinality += 1;
        }
        bool converged = true;
#pragma omp parallel for default(none) shared(newCentroids,converged,useStopCondition) num_threads(4)
        for (int i = 0; i < this->k; i++) {
            newCentroids[i].x /= newCentroids[i].cardinality;
            newCentroids[i].y /= newCentroids[i].cardinality;
            newCentroids[i].z /= newCentroids[i].cardinality;
            if (useStopCondition && std::abs(centroids[i].x - newCentroids[i].x) > 1e-15
                && std::abs(centroids[i].y - newCentroids[i].y) > 1e-15
                && std::abs(centroids[i].z - newCentroids[i].z) > 1e-15) {
                converged = false;
            }
        }

        if(useStopCondition && converged) {
            //std::cout << "stopped at iter: ";
            //std::cout << _ << std::endl;
            return;
        }

        this->centroids = newCentroids;
    }
}