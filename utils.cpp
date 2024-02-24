//
// Created by luigi on 18/02/24.
//
#include "utils.h"

std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}


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
        std::vector<std::string> coords = split(line,',');
        Point p;
        p.x = stod(coords[0]);
        p.y = stod(coords[1]);
        p.z = stod(coords[2]);
        points.push_back(p);
    }
    file.close();
    return points;
}


void exportCsv(const std::string& fileName, std::vector<Point>& dataPoints){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }
    for (Point& point : dataPoints) {
        file << point.toString() << "\n";
    }
    file.close();

}
