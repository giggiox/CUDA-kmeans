#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

#define K 2

#define THREAD_PER_BLOCK 512
#define A 1000 //alias for N set at compile time...

int N = 0;

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}



float* loadCsv(const std::string& fileName){
    std::string line;
    std::ifstream file(fileName);
    std::string word;
    if(!file.is_open()){
        std::cout << "error opening file." << std::endl;
        return nullptr;
    }
    int lineNumber = 0;
    while(getline(file,line)){
        lineNumber += 1;
    }
    N = lineNumber;
    file.close();

    std::ifstream file1(fileName);
    float *dataPoints = (float*) malloc(sizeof(float)*lineNumber*4);
    int i = 0;
    while(getline(file1,line)){
        std::vector<std::string> coords = split(line,',');
        dataPoints[i++] = stof(coords[0]);
		dataPoints[i++] = stof(coords[1]);
        dataPoints[i++] = stof(coords[2]);
		dataPoints[i++] = 0.0;
    }
    file1.close();
    return dataPoints;
}


void exportCsv(const std::string& fileName, float * dataPoints){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }
    for (int i = 0;i<N;i++) {
        file << dataPoints[4*i] << "," << dataPoints[4*i+1] << "," << dataPoints[4*i+2] << "," << dataPoints[4*i+3] << "\n";
    }
    file.close();

}

float *randomCentroids(float* dataPoints){
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(0,N-1);

    float * centroids = (float*) malloc(sizeof(float)*K*4);
    for(int i = 0; i < K; ++i){
        int index = dist6(rng);
        float x = dataPoints[index*4];
        float y = dataPoints[index*4+1];
        float z = dataPoints[index*4+2];
        centroids[4*i] = x;
        centroids[4*i+1] = y;
        centroids[4*i+2] = z;
        centroids[4*i+3]  = 0.0;
        
    }
    return centroids;
}



__device__ float euclideanDistance(float x1, float y1,float z1, float x2, float y2,float z2){
	return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1));
}

__global__ void centroidAssign(float *dataPoints_dev, float *centroids_dev){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= A) return;
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






__global__ void centroidUpdate1(float *dataPoints_dev,float *centroids_dev){
	

	const int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= 4*A) return;
	
	__shared__ float dataPoints_shared[THREAD_PER_BLOCK];
	dataPoints_shared[threadIdx.x]= dataPoints_dev[index];


	if(index < 4*K){centroids_dev[index] = 0.0;}


	__syncthreads();
	if(threadIdx.x==0){
		float xSum[K],ySum[K],zSum[K],clusterCardinality[K];

		for(int i = 0; i < blockDim.x; ++i){
            int clust_id = dataPoints_shared[4*i + 3];
            xSum[clust_id]+=dataPoints_shared[4*i];
			ySum[clust_id]+=dataPoints_shared[4*i+1];
            zSum[clust_id]+=dataPoints_shared[4*i+2];
			clusterCardinality[clust_id] += 1.0;
		}

		for(int i = 0;i < K; i++){
			atomicAdd(&centroids_dev[i*4],xSum[i]);
			atomicAdd(&centroids_dev[i*4+1],ySum[i]);
            atomicAdd(&centroids_dev[i*4+2],zSum[i]);
			atomicAdd(&centroids_dev[i*4+3],clusterCardinality[i]);
		}
	}
	__syncthreads();

	if(index < K){
		centroids_dev[index*4] = centroids_dev[index*4]/centroids_dev[index*4+3];
		centroids_dev[index*4+1] = centroids_dev[index*4+1]/centroids_dev[index*4+3];
        centroids_dev[index*4+2] = centroids_dev[index*4+2]/centroids_dev[index*4+3];  
		centroids_dev[index*4+3] = 0.0; 

	}


}

int main(int argc, char* argv[]){

    float *dataPoints = loadCsv("ciao.csv");
    N = 1000;
    float *centroids = randomCentroids(dataPoints);
	
	//float *dataPoints = (float*) malloc(sizeof(float)*3*N);
    //float *centroids = (float*)malloc(sizeof(float)*K*4);


	/*for(int i = 0;i <N ; i++){ std::cout << dataPoints[4*i] << " " << dataPoints[4*i+1] << " " << dataPoints[4*i+2] << " " << dataPoints[4*i+3] << std::endl;}
	std::cout << "ciao" << std::endl;
    */

    /*centroids[0] = 2.39014695;
	centroids[1] = -0.57684421;
	centroids[2] = -10;
    centroids[3] = 0.0;
    centroids[4] = -7.53619806;
	centroids[5] = 4.49955772;
	centroids[6] = 13.4;
    centroids[7] = 0.0;*/



    /*std::cout << "data points: " << std::endl;
    for(int i = 0; i < N; ++i){ std::cout << dataPoints[i*3] << std::endl;}
    std::cout << "centroids: " << std::endl;
    for(int i = 0; i < K; ++i){ std::cout << centroids[i*3] << std::endl;}
    std::cout << "end printing" << std::endl;*/
 

    float *dataPoints_dev;
	cudaMalloc(&dataPoints_dev, 4*N*sizeof(float));
    cudaMemcpy(dataPoints_dev,dataPoints,4*N*sizeof(float),cudaMemcpyHostToDevice);

    float *centroids_dev;
	cudaMalloc(&centroids_dev,4*K*sizeof(int));
    cudaMemcpy(centroids_dev,centroids,4*K*sizeof(float),cudaMemcpyHostToDevice);


    std::cout << (N*4+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK << std::endl;

    for (int i = 0;i<100;i++){
        centroidAssign<<<(N*4+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(dataPoints_dev,centroids_dev);


        
        cudaMemcpy(dataPoints,dataPoints_dev,4*N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids,centroids_dev,4*K*sizeof(float),cudaMemcpyDeviceToHost);
        /*std::cout << "cluster labels: " << std::endl;
        for(int i = 0; i < N; ++i){ std::cout << dataPoints[4*i+3] << ", ";}
        std::cout << "\nend printing" << std::endl;*/
        //cudaDeviceSynchronize();
        

        
        centroidUpdate1<<<(N*4+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK,THREAD_PER_BLOCK>>>(dataPoints_dev,centroids_dev);
        

        

    }
    cudaMemcpy(dataPoints,dataPoints_dev,4*N*sizeof(float),cudaMemcpyDeviceToHost);
    std::cout << "----------------------------cluster labels: -----------------" << std::endl;
    for(int i = 0; i < N; ++i){ std::cout << dataPoints[4*i+3] << ", ";}
    std::cout << "\n--------------------end printing-----------------------" << std::endl;

    exportCsv("fine.csv",dataPoints);


	return 0;
}
