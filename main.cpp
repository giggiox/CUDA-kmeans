#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "KMeans.h"
#include <omp.h>



int mainReportNumThreads(const std::string& dataset,const std::string& centroidsPath,const std::string& s_numthreads){
    int  numthreads= std::stoi(s_numthreads);
    std::vector<Point> data = loadDataset(dataset);
    KMeans kmean(5);
    std::vector<Centroid> centroids = loadCentroids(centroidsPath);
    kmean.centroids = centroids;
    double dt = omp_get_wtime();
    kmean.fitParallel(data,100,numthreads);
    dt = omp_get_wtime() - dt;
    std::cout << dt << std::endl;

    return 0;
}




int mainReport(const std::string& dataset,const std::string& centroidsPath){
    std::vector<Point> data = loadDataset(dataset);
    KMeans kmean(5);
    std::vector<Centroid> centroids = loadCentroids(centroidsPath);
    kmean.centroids = centroids;
    double dt = omp_get_wtime();
    kmean.fit(data,100);
    dt = omp_get_wtime() - dt;
    std::cout << dt << std::endl;

    kmean.centroids = centroids;
    dt = omp_get_wtime();
    kmean.fitParallel(data,100,4);
    dt = omp_get_wtime() - dt;
    std::cout << dt << std::endl;

    return 0;
}

int mainTest(){
    std::string cwd = "/home/luigi/CLionProjects/kmeans/";
    std::string dataset = "10000_5.csv";
    std::string centroidsPath = "10000_5_centroids.csv";

    std::vector<Point> data = loadDataset(cwd + "dataset/" + dataset);
    std::vector<Centroid> centroids = loadCentroids(cwd + "dataset/" + centroidsPath);

    KMeans kmean(5);
    kmean.centroids = centroids;

    double dt = omp_get_wtime();
    kmean.fit(data,100);
    dt = omp_get_wtime() - dt;
    std::cout << "time sequential execution: " << dt << std::endl;


    std::cout << "found centroids sequential" << std::endl;
    for(int i = 0; i < kmean.centroids.size();++i){
        std::cout << kmean.centroids[i].toString() << std::endl;
    }


    kmean.centroids = centroids;
    dt = omp_get_wtime();
    kmean.fitParallel(data,100,4);
    dt = omp_get_wtime() - dt;
    std::cout << "time parallel execution: " << dt << std::endl;


    std::cout << "found centroids parallel" << std::endl;
    for(int i = 0; i < kmean.centroids.size();++i){
        std::cout << kmean.centroids[i].toString() << std::endl;
    }




    return 0;
}



int main(int argc, char* argv[]) {
    if(argc == 1) {
        return mainTest();
    }else if(argc == 3){
        return mainReport(argv[1],argv[2]);
    }else if(argc == 4) {
        return mainReportNumThreads(argv[1],argv[2],argv[3]);
    }else{
        std::cerr << "Pass the correct amount of arguments." << std::endl;
        return 1;
    }
}

