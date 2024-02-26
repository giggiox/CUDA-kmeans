#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

#define K 2


#define A 10 //alias per N

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
    float *dataPoints = (float*) malloc(sizeof(float)*lineNumber);
    int i = 0;
    while(getline(file1,line)){
        std::vector<std::string> coords = split(line,',');
        dataPoints[i++] = stof(coords[0]);
    }
    file1.close();
    return dataPoints;
}


__device__ float euclideanDistance(float x1, float x2){
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void centroidAssign(float *dataPoints_dev, float *centroids_dev, int *clusterLabels_dev){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (index >= A) return;
	float minDistance = INFINITY;
	int clusterLabel = 0;


	for(int j = 0; j < K; ++j){
		float distance = euclideanDistance(dataPoints_dev[index],centroids_dev[j]);
        //printf("index: %d distance: %f j: %d  dataPoint: %f centroid: %f \n",index,distance,j,dataPoints_dev[index],centroids_dev[j]);
		if(distance < minDistance){
			minDistance = distance;
			clusterLabel = j;
		}
	}
	clusterLabels_dev[index]=clusterLabel;
}


float *randomCentroids(float* dataPoints){
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(0,N-1);

    float * centroids = (float*) malloc(sizeof(float)*K);
    for(int i = 0; i < K; ++i){
        int index = dist6(rng);
        float x = dataPoints[index];
        centroids[i] = x;
        
    }
    return centroids;
}



int main(int argc, char* argv[]){

    float *dataPoints = loadCsv("10.csv");
    //float *centroids = randomCentroids(dataPoints);
    float *centroids = (float*)malloc(sizeof(float)*K);
    centroids[0] = 23.2;
    centroids[1] = 2.3;

    std::cout << "data points: " << std::endl;
    for(int i = 0; i < N; ++i){ std::cout << dataPoints[i] << std::endl;}
    std::cout << "centroids: " << std::endl;
    for(int i = 0; i < K; ++i){ std::cout << centroids[i] << std::endl;}
    std::cout << "end printing" << std::endl;
 

    float *dataPoints_dev;
	cudaMalloc(&dataPoints_dev, N*sizeof(float));
    cudaMemcpy(dataPoints_dev,dataPoints,N*sizeof(float),cudaMemcpyHostToDevice);

    float *centroids_dev;
	cudaMalloc(&centroids_dev,K*sizeof(int));
    cudaMemcpy(centroids_dev,centroids,K*sizeof(float),cudaMemcpyHostToDevice);

    int *clusterLabels_dev = 0;
    cudaMalloc(&clusterLabels_dev,N*sizeof(float));



    centroidAssign<<<2,5>>>(dataPoints_dev,centroids_dev,clusterLabels_dev);

    
    int *clusterLabels = (int*) malloc(sizeof(int)*N);
    cudaMemcpy(clusterLabels,clusterLabels_dev,N*sizeof(int),cudaMemcpyDeviceToHost);

    std::cout << "cluster labels: " << std::endl;
    for(int i = 0; i < N; ++i){ std::cout << clusterLabels[i] << ", ";}
    std::cout << "\nend printing" << std::endl;


	return 0;
}