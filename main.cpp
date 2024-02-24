#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "KMeans.h"
#include <omp.h>


#define USESTOPCONDITION false

int mainReport(const std::string& dataset){
    std::vector<Point> data = loadCsv(dataset);
    KMeans kmean(5);
    kmean.assignRandomCentroids(data);
    std::vector<Centroid> centroids = kmean.centroids;
    double dt = omp_get_wtime();
    kmean.fit(data,100, USESTOPCONDITION);
    dt = omp_get_wtime() - dt;
    std::cout << dt << std::endl;

    kmean.centroids = centroids;
    dt = omp_get_wtime();
    kmean.fitParallel(data,100, USESTOPCONDITION);
    dt = omp_get_wtime() - dt;
    std::cout << dt << std::endl;

    return 0;
}

int mainTest(){
    std::string cwd = "/home/luigi/CLionProjects/kmeans/";
    std::string dataset = "1000_5.csv";

    std::vector<Point> data = loadCsv(cwd + "dataset/" + dataset);

    KMeans kmean(5);
    kmean.assignRandomCentroids(data);

    std::cout << "centroid initialization" << std::endl;

    std::cout << "centroids" << std::endl;
    for(int i = 0;i<kmean.centroids.size();++i){
        std::cout << kmean.centroids[i].toString() << std::endl;
    }



    double dt = omp_get_wtime();
    kmean.fit(data,100, false);
    dt = omp_get_wtime() - dt;
    std::cout << "time sequential execution: " << dt << std::endl;
    exportCsv(cwd + "result/" + dataset,data);


    std::vector<Centroid> centroids = kmean.centroids;
    std::cout << "found centroids sequential" << std::endl;
    for(int i = 0; i < centroids.size();++i){
        std::cout << centroids[i].toString() << std::endl;
    }

    return 0;
}



int main(int argc, char* argv[]) {
    if(argc == 1) {
        return mainTest();
    }else if(argc == 2){
        return mainReport(argv[1]);
    }else{
        std::cerr << "Pass the correct amount of arguments." << std::endl;
        return 1;
    }
}

