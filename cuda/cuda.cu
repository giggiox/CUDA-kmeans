#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
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


__device__ float euclideanDistance(float x1, float y1,float z1, float x2, float y2,float z2){
    return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
}

__global__ void centroidAssign(float *dataPoints_dev, float *centroids_dev,int N){
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= N) return;
    float minDistance = INFINITY;
    int clusterLabel = 0;
    for(int j = 0; j < K; ++j){
        float distance = euclideanDistance(dataPoints_dev[index*4],dataPoints_dev[index*4+1],dataPoints_dev[index*4+2],centroids_dev[j*4],centroids_dev[j*4+1],centroids_dev[j*4+2]);
        if(distance < minDistance){
            minDistance = distance;
            clusterLabel = j;
        }
    }
    dataPoints_dev[index*4+3]=clusterLabel;
}

__global__ void centroidUpdate(float *dataPoints_dev,float *centroids_dev,int N){
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= 4*N) {return;}
    __shared__ float dataPoints_shared[THREAD_PER_BLOCK];
    dataPoints_shared[threadIdx.x]= dataPoints_dev[index];
    if(index < 4*K){ centroids_dev[index] = 0.0;}

    __syncthreads();

    if(threadIdx.x==0){
        float xSum[K] = {0.0},ySum[K]= {0.0},zSum[K]= {0.0},clusterCardinality[K]= {0.0};
        for(int i = 0; i < THREAD_PER_BLOCK/4 -1; ++i){
            if (blockIdx.x != (gridDim.x-1) || i < ((4*N)-(blockDim.x*(gridDim.x-1)))/4-1){
                int clust_id = (int) dataPoints_shared[4*i+ 3];
                xSum[clust_id]+=dataPoints_shared[4*i];
                ySum[clust_id]+=dataPoints_shared[4*i+1];
                zSum[clust_id]+=dataPoints_shared[4*i+2];
                clusterCardinality[clust_id] += 1.0;
            }
        }

        for(int i = 0;i < K; i++){
            atomicAdd(&(centroids_dev[i*4]),xSum[i]);
            atomicAdd(&(centroids_dev[i*4+1]),ySum[i]);
            atomicAdd(&(centroids_dev[i*4+2]),zSum[i]);
            atomicAdd(&(centroids_dev[i*4+3]),clusterCardinality[i]);
        }
    }

}






int main(int argc, char* argv[]){
    if(argc < 3) return -1;
    int N = getLineNumber(argv[1]);
    float *dataPoints = loadCsv(argv[1]);
    float *centroids = loadCsv(argv[2]);

    auto wcts = std::chrono::system_clock::now(); //wall clock

    float *dataPoints_dev;

    CUDA_CHECK_RETURN(cudaMalloc(&dataPoints_dev, 4*N*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(dataPoints_dev,dataPoints,4*N*sizeof(float),cudaMemcpyHostToDevice));

    float *centroids_dev;
    CUDA_CHECK_RETURN(cudaMalloc(&centroids_dev,4*K*sizeof(float)));
    CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev,centroids,4*K*sizeof(float),cudaMemcpyHostToDevice));


    const int clustSize = (4*N+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK;
    const int blockSize = THREAD_PER_BLOCK;

    //std::cout << "using: " << clustSize << " " << blockSize << std::endl;

    for( int iter = 0; iter < 100; ++iter){
        cudaDeviceSynchronize();
        centroidAssign<<<clustSize,blockSize>>>(dataPoints_dev,centroids_dev,N);
        cudaMemset(centroids_dev,0.0,sizeof(float)*K*4);
        cudaDeviceSynchronize();
        centroidUpdate<<<clustSize,blockSize>>>(dataPoints_dev,centroids_dev,N);
        cudaDeviceSynchronize();
        CUDA_CHECK_RETURN(cudaMemcpy(centroids,centroids_dev,4*K*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i = 0;i<K;i++){
            int cardinality = (int) centroids[4*i+3];
            centroids[4*i] = centroids[4*i]/cardinality;
            centroids[4*i+1] = centroids[4*i+1]/cardinality;
            centroids[4*i+2] = centroids[4*i+2]/cardinality;
        }
        CUDA_CHECK_RETURN(cudaMemcpy(centroids_dev,centroids,4*K*sizeof(float),cudaMemcpyHostToDevice));

        //cudaDeviceSynchronize();
        //uncomment to see how centroids are updated after each iteration
        /*CUDA_CHECK_RETURN(cudaMemcpy(centroids,centroids_dev,4*K*sizeof(float),cudaMemcpyDeviceToHost));
        std::cout << "Centroids: " << std::endl;
        for(int i = 0; i < K;i++){ std::cout << centroids[4*i] << ","<< centroids[4*i+1] << "," <<centroids[4*i+2] << "," << centroids[4*i+3] << std::endl;}
        */
    }


    CUDA_CHECK_RETURN(cudaMemcpy(centroids,centroids_dev,4*K*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(dataPoints,dataPoints_dev,4*N*sizeof(float),cudaMemcpyDeviceToHost));


    std::chrono::duration<float> wctduration = (std::chrono::system_clock::now() - wcts);
    std::cout << wctduration.count() << std::endl;

    /* uncomment to see final centroids
    std::cout << "Centroids: " << std::endl;
    for(int i = 0; i < K;i++){ std::cout << centroids[4*i] << ","<< centroids[4*i+1] << "," <<centroids[4*i+2] << "," << centroids[4*i+3] << std::endl;}
    */

    //exportCsv("result.csv",dataPoints,N);

    free(dataPoints); free(centroids);
    cudaFree(dataPoints_dev); cudaFree(centroids_dev);


    return 0;
}