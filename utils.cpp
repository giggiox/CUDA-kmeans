//
// Created by luigi on 18/02/24.
//
#include "utils.h"

std::vector<Point> loadCsv(const std::string& fileName){
    std::vector<Point> points;
    std::string line;
    std::ifstream file(fileName);
    std::string word;
    if(!file.is_open()){
        std::cout << "error opening file." << std::endl;
        return points;
    }
    while(getline(file,line)){
        std::stringstream str(line);
        Point p; int i = 0;
        while(getline(str, word, ',')){
            p.coords[i] = std::stod(word);
            i++;
        }
        points.push_back(p);
    }
    return points;
}


void exportCsv(const std::string& fileName, std::vector<Point>& dataPoints){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "error opening file." << std::endl;
        return;
    }
    for (const auto& point : dataPoints) {
        for (const auto& coord : point.coords) {
            file << coord << ",";
        }
        file << point.clusterLabel << "\n";
    }
    file.close();

}

