#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include "utils.h"

#define K 5
#define THREAD_PER_BLOCK 1024

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement << " returned " << cudaGetErrorString(err)
              << " (" << err << ") at " << file << ":" << line << std::endl;
    exit(1);
}

__device__ float distanceMetric(float x1, float y1, float x2, float y2) {
    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

__global__ void assignmentKernel(const float *dataPoints, const float *centroids,
                                 int *clusterLabels,
                                 float *partialCentroids,
                                 int   *partialCardinalities,
                                 int N) {
    int globalIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalIndex >= N) return;

    int localIndex  = threadIdx.x;
    
    __shared__ float sharedCentroids[2 * K];
    __shared__ int sharedCardinalities[K];
    
    for (int i = localIndex; i < 2 * K; i += blockDim.x) {
        sharedCentroids[i] = 0.0f;
        if (i < K)
            sharedCardinalities[i] = 0;
    }
    __syncthreads();

    float minDistance = INFINITY;
    int clusterLabel = 0;
    for (int j = 0; j < K; ++j) {
        float distance = distanceMetric(dataPoints[globalIndex*2], dataPoints[globalIndex*2+1],
                                        centroids[j*2], centroids[j*2+1]);
        if(distance < minDistance){
            minDistance = distance;
            clusterLabel = j;
        }
    }
    clusterLabels[globalIndex] = clusterLabel;
    atomicAdd(&(sharedCentroids[clusterLabel*2]), dataPoints[globalIndex*2]);
    atomicAdd(&(sharedCentroids[clusterLabel*2 + 1]), dataPoints[globalIndex*2+1]);
    atomicAdd(&(sharedCardinalities[clusterLabel]), 1);

    __syncthreads();

    for (int i = localIndex; i < K; i += blockDim.x) {
        int offset_cent = blockIdx.x * K * 2;
        int offset_card = blockIdx.x * K;
        partialCentroids[offset_cent + i * 2] = sharedCentroids[i * 2];
        partialCentroids[offset_cent + i * 2 + 1] = sharedCentroids[i * 2 + 1];
        partialCardinalities[offset_card + i] = sharedCardinalities[i];
    }
}

__global__ void reductionKernel(const float *partialCentroids,
                                const int   *partialCardinalities,
                                float *finalCentroids,     // Dimensione: K * 2
                                int   *finalCardinalities,   // Dimensione: K
                                int numBlocks) {
    int clusterIdx = threadIdx.x;
    if (clusterIdx >= K) return;
    
    float sumX = 0.0f;
    float sumY = 0.0f;
    int totalCount = 0;
    
    for (int b = 0; b < numBlocks; ++b) {
        int offset_cent = b * K * 2;
        int offset_card = b * K;
        sumX += partialCentroids[offset_cent + clusterIdx * 2];
        sumY += partialCentroids[offset_cent + clusterIdx * 2 + 1];
        totalCount += partialCardinalities[offset_card + clusterIdx];
    }
    finalCentroids[clusterIdx * 2]     = sumX;
    finalCentroids[clusterIdx * 2 + 1] = sumY;
    finalCardinalities[clusterIdx]     = totalCount;
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Usage: ./kmeansCuda <dataPoints.csv> <centroids.csv>" << std::endl;
        return -1;
    }
    
    int N = getLineNumber(argv[1]);
    float *dataPoints = loadCsv(argv[1]);
    float *centroids  = loadCsv(argv[2]);
    
    float *dataPoints_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&dataPoints_dev, 2 * N * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(dataPoints_dev, dataPoints, 2 * N * sizeof(float), cudaMemcpyHostToDevice));
    
    float *centroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&centroids_dev, 2 * K * sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, 2 * K * sizeof(float), cudaMemcpyHostToDevice));
    
    int *clusterLabels = (int*) malloc(N * sizeof(int));
    int *clusterLabels_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&clusterLabels_dev, N * sizeof(int)));
    
    float *newCentroids = (float*) malloc(K * 2 * sizeof(float));
    int   *clusterCardinalities = (int*) malloc(K * sizeof(int));
    
    float *finalCentroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&finalCentroids_dev, K * 2 * sizeof(float)));
    int *finalCardinalities_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&finalCardinalities_dev, K * sizeof(int)));
    
    int gridSize = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    float *partialCentroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&partialCentroids_dev, gridSize * K * 2 * sizeof(float)));
    int *partialCardinalities_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&partialCardinalities_dev, gridSize * K * sizeof(int)));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < 100; ++iter) {
        CUDA_CHECK_RETURN(cudaMemset(partialCentroids_dev, 0, gridSize * K * 2 * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemset(partialCardinalities_dev, 0, gridSize * K * sizeof(int)));
        
        assignmentKernel<<<gridSize, THREAD_PER_BLOCK>>>(dataPoints_dev, centroids_dev, clusterLabels_dev,
                                                           partialCentroids_dev, partialCardinalities_dev, N);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        
        reductionKernel<<<1, K>>>(partialCentroids_dev, partialCardinalities_dev,
                                  finalCentroids_dev, finalCardinalities_dev, gridSize);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        
        CUDA_CHECK_RETURN(cudaMemcpy(newCentroids, finalCentroids_dev, K * 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(clusterCardinalities, finalCardinalities_dev, K * sizeof(int), cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < K; ++i) {
            int count = clusterCardinalities[i];
            if (count <= 0) continue;
            centroids[i * 2]     = newCentroids[i * 2]     / count;
            centroids[i * 2 + 1] = newCentroids[i * 2 + 1] / count;
        }
        CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK_RETURN(cudaMemcpy(clusterLabels, clusterLabels_dev, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end_time - start_time;
    std::cout << duration.count()  << std::endl;
    
#ifdef PRINT_FINAL_CENTROIDS
    std::cout << "Final centroids:" << std::endl; 
    for (int i = 0; i < K; i++) {
        std::cout << centroids[i * 2] << ", " << centroids[i * 2 + 1] << std::endl;
    }
#endif    

#ifdef EXPORT_FINAL_RESULT
    exportCsv("path/cudares.csv", dataPoints, clusterLabels, N);
    std::cout << "Done exporting result to csv." << std::endl; 
#endif

    
    free(dataPoints); cudaFree(dataPoints_dev);
    free(centroids); cudaFree(centroids_dev);
    free(newCentroids); cudaFree(partialCentroids_dev); cudaFree(finalCentroids_dev);
    free(clusterLabels); cudaFree(clusterLabels_dev);
    free(clusterCardinalities); cudaFree(partialCardinalities_dev); cudaFree(finalCardinalities_dev);
    
    return 0;
}
