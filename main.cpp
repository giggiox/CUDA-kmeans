#include <iostream>
#include "KMeans.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    //KMeans kmean(3);
    //kmean.fit();
    double a[2] = {2.0,2.0};
    Point p1 = Point(a);

    std::cout << p1.coords[0] << std::endl;
    std::cout << p1.coords[1] << std::endl;
    std::cout << p1.toString() << std::endl;
    return 0;
}
