#include <iostream>
#include <fstream>
#include <sstream>
#include "KMeans.h"



std::vector<Point> loadCsv(const std::string& file_name){
    std::vector<Point> points;
    std::string line;
    std::ifstream file(file_name);
    std::string word;
    if(!file.is_open()){
        std::cout << "can't open file" << std::endl;
        return points;
    }
    while(getline(file,line)){
        std::stringstream str(line);
        Point p; int i = 0;
        while(getline(str, word, ',')){
            p.coords[i] = std::stod(word);
            i++;
        }
        p.clusterLabel = -1;
        points.push_back(p);
    }
    return points;
}


int main() {
    std::vector<Point> data = loadCsv("/home/luigi/CLionProjects/ompkmeans/data.csv");
    /**for(int i = 0;i<data.size();++i){
        std::cout << data[i].toString() << std::endl;
    }**/


    KMeans kmean(3);
    kmean.fit(data,100);
    std::vector<Centroid> centroids = kmean.centroids;
    for(int i = 0; i< centroids.size();++i){
        std::cout << centroids[i].toString() << std::endl;
    }

    //double a[2] = {2.0,2.0};
    //Point p1 = Point(a);
    //std::cout << p1.coords[0] << std::endl;
    //std::cout << p1.coords[1] << std::endl;
    //std::cout << p1.toString() << std::endl;
    return 0;
}
