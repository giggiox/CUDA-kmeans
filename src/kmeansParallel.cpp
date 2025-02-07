#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include "utils.h"

#define K 5

float distanceMetric(float x1, float y1, float x2, float y2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1); // Squared Euclidean distance
}


void centroidAssignAndUpdate(
    float* dataPoints, float* centroids, float* newCentroids, int* clusterCardinality, 
    int* clusterLabel, int N) {


#pragma omp parallel
{
    float newCentroidsLocal[K * 2] = {0.0f};
    int clusterCardinalityLocal[K] = {0};

#pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) {
        float minDistance = INFINITY;
        int cluster = 0;
        for (int j = 0; j < K; ++j) {
            float distance = distanceMetric(dataPoints[i * 2], dataPoints[i * 2 + 1], centroids[j * 2], centroids[j * 2 + 1]);
            if (distance < minDistance) {
                minDistance = distance;
                cluster = j;
            }
        }
        clusterLabel[i] = cluster;
        newCentroidsLocal[cluster * 2] += dataPoints[i * 2];
        newCentroidsLocal[cluster * 2 + 1] += dataPoints[i * 2 + 1];
        clusterCardinalityLocal[cluster]++;
    }

    for (int i = 0; i < K; ++i) {
#pragma omp atomic
        clusterCardinality[i] += clusterCardinalityLocal[i];
#pragma omp atomic
        newCentroids[2*i] += newCentroidsLocal[2*i];
#pragma omp atomic
        newCentroids[2*i+1] += newCentroidsLocal[2*i+1];
    }

}

#pragma omp parallel for schedule(static)
    for (int i = 0; i < K; ++i) {
        if (clusterCardinality[i] <= 0) continue;
        newCentroids[i * 2] = newCentroids[i * 2] / clusterCardinality[i];
        newCentroids[i * 2 + 1] = newCentroids[i * 2 + 1] / clusterCardinality[i];
    }
}

int main(int argc, char const *argv[]) {
    if (argc < 3) return -1;

    int N = getLineNumber(argv[1]);
    float* dataPoints = loadCsv(argv[1]);
    float* centroids = loadCsv(argv[2]);

    auto wcts = std::chrono::system_clock::now(); // Wall clock start

    int* clusterLabel = (int*) malloc(N * sizeof(int));
    int* clusterCardinality = (int*) malloc(K * sizeof(int));

    float* newCentroids = (float*) malloc(K * 2 * sizeof(float));
    for (int iter = 0; iter < 100; ++iter) {
        std::memset(newCentroids, 0.0f, K * 2 * sizeof(float));
        std::memset(clusterCardinality, 0, K * sizeof(int));

        centroidAssignAndUpdate(dataPoints, centroids, newCentroids, clusterCardinality, clusterLabel, N);

        memcpy(centroids, newCentroids, K * 2 * sizeof(float));
    }

    std::chrono::duration<float> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << wctduration.count() << std::endl;

#ifdef PRINT_FINAL_CENTROIDS
    std::cout << "Final centroids: " << std::endl; 
    for(int i=0;i<K; ++i){ std::cout << centroids[i*2] << ", " << centroids[i*2+1] << std::endl; }
#endif    

#ifdef EXPORT_FINAL_RESULT
    exportCsv("path/cudares.csv",dataPoints, clusterLabel, N);
    std::cout << "Done exporting result to csv. " << std::endl; 
#endif

    free(dataPoints); free(centroids);
    free(clusterLabel); free(clusterCardinality);
    free(newCentroids); 

    return 0;
}
