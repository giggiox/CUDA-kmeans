//
// Created by luigi on 18/02/24.
//

#ifndef OMPKMEANS_UTILS_H
#define OMPKMEANS_UTILS_H
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "point.h"
#include "centroid.h"

std::vector<std::string> split (const std::string &s, char delim);
void exportCsv(const std::string& fileName, std::vector<Point>& dataPoints);
std::vector<Point> loadDataset(const std::string& fileName);
std::vector<Centroid> loadCentroids(const std::string& fileName);

#endif //OMPKMEANS_UTILS_H