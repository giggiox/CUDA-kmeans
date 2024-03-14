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




void KMeans::fit(std::vector<Point>& dataPoints, int maxIteration) {
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
        for (int i = 0; i < this->k; i++) {
            newCentroids[i].x /= newCentroids[i].cardinality;
            newCentroids[i].y /= newCentroids[i].cardinality;
            newCentroids[i].z /= newCentroids[i].cardinality;
        }
        this->centroids = newCentroids;
    }
}


void KMeans::fitParallel(std::vector<Point>& dataPoints, int maxIteration) {
    if(this->centroids.empty()){
        std::cerr << "assign centroids first" << std::endl;
        return;
    }

    for (int _ = 0; _ < maxIteration; ++_) {
            std::vector<Centroid> newCentroids = std::vector<Centroid>(this->k, Centroid());
#pragma omp parallel
        {
            std::vector<Centroid> newCentroidsTmp = std::vector<Centroid>(this->k, Centroid());
#pragma omp for schedule(static)
            for (int i = 0; i < dataPoints.size(); ++i) {
                Point &p = dataPoints[i];
                float minDistance = INFINITY;

                for (int j = 0; j < this->k; ++j) {
                    float distance = euclideanNorm(p, this->centroids[j]);
                    if (distance < minDistance) {
                        minDistance = distance;
                        p.clusterLabel = j;
                    }
                }
                newCentroidsTmp[p.clusterLabel].cardinality += 1;
                newCentroidsTmp[p.clusterLabel].x += dataPoints[i].x;
                newCentroidsTmp[p.clusterLabel].y += dataPoints[i].y;
                newCentroidsTmp[p.clusterLabel].z += dataPoints[i].z;
            }

            for (int j = 0; j < this->k; ++j) {
#pragma omp atomic
                newCentroids[j].cardinality += newCentroidsTmp[j].cardinality;
#pragma omp atomic
                newCentroids[j].x += newCentroidsTmp[j].x;
#pragma omp atomic
                newCentroids[j].y += newCentroidsTmp[j].y;
#pragma omp atomic
                newCentroids[j].z += newCentroidsTmp[j].z;
            }

        }
        for (int i = 0; i < this->k; ++i) {
            newCentroids[i].x /= newCentroids[i].cardinality;
            newCentroids[i].y /= newCentroids[i].cardinality;
            newCentroids[i].z /= newCentroids[i].cardinality;
        }

        this->centroids = newCentroids;
    }
}


