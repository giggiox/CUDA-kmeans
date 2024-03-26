#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>
#include "utils.h"

#define K 5
#define THREAD_PER_BLOCK 1024


#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__,#value, value)
static void CheckCudaErrorAux (const char *file, unsigned line, const char*statement, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::cerr << statement<<" returned " << cudaGetErrorString(err) <<"("<<err<< ") at "<<file<<":"<<line << std::endl;
    exit (1);
}

__device__ float distanceMetric(float x1, float y1,float z1, float x2, float y2,float z2){
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1); //squared euclidean distance
}

__global__ void centroidAssignAndUpdate(float *dataPoints_dev,  float *centroids_dev, float *newCentroids_dev, int *clusterCardinality_dev,int*clusterLabel_dev, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= N) {return;}
    int localIndex = threadIdx.x;
    __shared__ float newCentroids_shared[3*K];
    __shared__ int clusterCardinality_shared[K];
    if(localIndex < 3*K) { newCentroids_dev[localIndex] = 0.0; newCentroids_shared[localIndex] = 0.0;}
    if(localIndex < K) { clusterCardinality_dev[localIndex] = 0; clusterCardinality_shared[localIndex] = 0; }

    __syncthreads();
    float minDistance = INFINITY;
    int clusterLabel = 0;
    for (int j = 0; j < K; j++) {
        float distance = distanceMetric(dataPoints_dev[index*3],dataPoints_dev[index*3+1],dataPoints_dev[index*3+2],centroids_dev[j*3],centroids_dev[j*3+1],centroids_dev[j*3+2]);
        if(distance < minDistance){
            minDistance = distance;
            clusterLabel = j;
        }
    }
    clusterLabel_dev[index] = clusterLabel;
    atomicAdd(&(newCentroids_shared[clusterLabel*3]), dataPoints_dev[index*3]);
    atomicAdd(&(newCentroids_shared[clusterLabel*3 + 1]), dataPoints_dev[index*3 + 1]);
    atomicAdd(&(newCentroids_shared[clusterLabel*3 + 2]), dataPoints_dev[index*3 + 2]);
    atomicAdd(&(clusterCardinality_shared[clusterLabel]),1);
    __syncthreads();

    if(localIndex < K) {
        atomicAdd(&(newCentroids_dev[localIndex*3]), newCentroids_shared[localIndex*3]);
        atomicAdd(&(newCentroids_dev[localIndex*3+1]), newCentroids_shared[localIndex*3+1]);
        atomicAdd(&(newCentroids_dev[localIndex*3+2]), newCentroids_shared[localIndex*3+2]);
        atomicAdd(&(clusterCardinality_dev[localIndex]), clusterCardinality_shared[localIndex]);
    }
}


int main(int argc, char const *argv[]) {
    if(argc < 3) return -1;
    int N = getLineNumber(argv[1]);
    float *dataPoints = loadCsv(argv[1]);
    float *centroids = loadCsv(argv[2]);

    auto wcts = std::chrono::system_clock::now(); //wall clock


    float *dataPoints_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&dataPoints_dev, 3*N*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(dataPoints_dev, dataPoints, N*3*sizeof(float), cudaMemcpyHostToDevice));

    float *centroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&centroids_dev, 3*K*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, K*3*sizeof(float), cudaMemcpyHostToDevice));

    int *clusterLabel = (int*) malloc(sizeof(int)*N);
    int *clusterLabel_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&clusterLabel_dev, N*sizeof(int)));

    float *newCentroids = (float*)malloc(sizeof(float)*K*3);
    float *newCentroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&newCentroids_dev, K*3*sizeof(float)));

    int * clusterCardinality = (int*) malloc(sizeof(int)*K);
    int *clusterCardinality_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&clusterCardinality_dev, K*sizeof(int)));

    const int gridSize = (N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;
    const int blockSize = THREAD_PER_BLOCK;

    for(int iter = 0; iter < 100; ++iter){
        centroidAssignAndUpdate<<<gridSize, blockSize>>>(dataPoints_dev,centroids_dev,newCentroids_dev,clusterCardinality_dev,clusterLabel_dev,N);
        CUDA_CHECK_RETURN(cudaMemcpy(newCentroids, newCentroids_dev, K*3*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(clusterCardinality, clusterCardinality_dev, K*sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < K; ++i) {
            int cardinality = clusterCardinality[i];
            centroids[i*3] = newCentroids[i*3] / cardinality;
            centroids[i*3+1] = newCentroids[i*3+1] / cardinality;
            centroids[i*3+2] = newCentroids[i*3+2] / cardinality;
        }
        CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, K*3*sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK_RETURN(cudaMemcpy(clusterLabel, clusterLabel_dev, N*sizeof(int), cudaMemcpyDeviceToHost));

    std::chrono::duration<float> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << wctduration.count() << std::endl;

    //uncomment to see final centroids
    /*
    std::cout << "final centroids: " << std::endl;
    for(int i=0;i<K;i++){std::cout << centroids[i*3] << ", " << centroids[i*3+1] << ", " << centroids[i*3+2] << std::endl;}
    */

    //uncomment to export dataset with updated cluster labels (visualize results with visualize_results.py)
    //exportCsv("result/cudares.csv",dataPoints,clusterLabel,N);

    free(dataPoints); free(centroids);
    free(newCentroids); free(clusterCardinality);
    cudaFree(dataPoints);cudaFree(centroids_dev);
    cudaFree(newCentroids_dev);cudaFree(clusterCardinality_dev);


    return 0;
}
