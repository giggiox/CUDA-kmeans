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

__device__ float distanceMetric(float x1, float y1, float x2, float y2){
    return (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1); // Squared euclidean distance
}

__global__ void centroidAssignAndUpdate(float *dataPoints_dev,  float *centroids_dev, float *newCentroids_dev, int *clusterCardinality_dev,int*clusterLabel_dev, int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= N) {return;}
    int localIndex = threadIdx.x;
    __shared__ float newCentroids_shared[2*K];
    __shared__ int clusterCardinality_shared[K];

    for(int i = localIndex; i < 2*K; i += blockDim.x) {
        newCentroids_shared[i] = 0.0;
        if (i < K) {
            clusterCardinality_shared[i] = 0;
        }
    }

    __syncthreads();
    float minDistance = INFINITY;
    int clusterLabel = 0;
    for (int j = 0; j < K; ++j) {
        float distance = distanceMetric(dataPoints_dev[index*2],dataPoints_dev[index*2+1],
                                        centroids_dev[j*2],centroids_dev[j*2+1]);
        if(distance < minDistance){
            minDistance = distance;
            clusterLabel = j;
        }
    }
    clusterLabel_dev[index] = clusterLabel;
    atomicAdd(&(newCentroids_shared[clusterLabel*2]), dataPoints_dev[index*2]);
    atomicAdd(&(newCentroids_shared[clusterLabel*2 + 1]), dataPoints_dev[index*2 + 1]);
    atomicAdd(&(clusterCardinality_shared[clusterLabel]),1);
    __syncthreads();

    for(int i = localIndex; i < K; i+= blockDim.x) {
        atomicAdd(&(newCentroids_dev[i*2]), newCentroids_shared[i*2]);
        atomicAdd(&(newCentroids_dev[i*2+1]), newCentroids_shared[i*2+1]);
        atomicAdd(&(clusterCardinality_dev[i]), clusterCardinality_shared[i]);
    }
}


int main(int argc, char const *argv[]) {
    if(argc < 3) return -1;
    int N = getLineNumber(argv[1]);
    float *dataPoints = loadCsv(argv[1]);
    float *centroids = loadCsv(argv[2]);

    auto wcts = std::chrono::system_clock::now(); // Wall clock

    float *dataPoints_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&dataPoints_dev, 2*N*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(dataPoints_dev, dataPoints, N*2*sizeof(float), cudaMemcpyHostToDevice));

    float *centroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&centroids_dev, 2*K*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, K*2*sizeof(float), cudaMemcpyHostToDevice));

    int *clusterLabel = (int*) malloc(sizeof(int)*N);
    int *clusterLabel_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&clusterLabel_dev, N*sizeof(int)));

    float *newCentroids = (float*)malloc(sizeof(float)*K*2);
    float *newCentroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&newCentroids_dev, K*2*sizeof(float)));

    int * clusterCardinality = (int*) malloc(sizeof(int)*K);
    int *clusterCardinality_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&clusterCardinality_dev, K*sizeof(int)));

    const int gridSize = (N + THREAD_PER_BLOCK - 1)/THREAD_PER_BLOCK;
    const int blockSize = THREAD_PER_BLOCK;

    for(int iter = 0; iter < 100; ++iter){
        CUDA_CHECK_RETURN(cudaMemset(newCentroids_dev, 0.0f, 2 * K * sizeof(float)));
        CUDA_CHECK_RETURN(cudaMemset(clusterCardinality_dev, 0, K * sizeof(int)));

        centroidAssignAndUpdate<<<gridSize, blockSize>>>(dataPoints_dev,centroids_dev,newCentroids_dev,clusterCardinality_dev,clusterLabel_dev,N);

        CUDA_CHECK_RETURN(cudaMemcpy(newCentroids, newCentroids_dev, K*2*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(clusterCardinality, clusterCardinality_dev, K*sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < K; ++i) {
            int cardinality = clusterCardinality[i];
            if (cardinality <= 0) continue; 
            centroids[i*2] = newCentroids[i*2] / cardinality;
            centroids[i*2+1] = newCentroids[i*2+1] / cardinality;
        }
        CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev, centroids, K*2*sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK_RETURN(cudaMemcpy(clusterLabel, clusterLabel_dev, N*sizeof(int), cudaMemcpyDeviceToHost));

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
    free(newCentroids); free(clusterCardinality);
    cudaFree(dataPoints); cudaFree(centroids_dev);
    cudaFree(newCentroids_dev); cudaFree(clusterCardinality_dev);

    return 0;
}
