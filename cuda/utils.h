std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}


int getLineNumber(const std::string& fileName){
    std::string line;
    std::ifstream file(fileName);
    std::string word;
    if(!file.is_open()){
        std::cout << "error opening file." << std::endl;
        return -1;
    }
    int lineNumber = 0;
    while(getline(file,line)){
        lineNumber += 1;
    }
    file.close();
    return lineNumber;
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


void exportCsv(const std::string& fileName, float * dataPoints,int dataPointsLength){
    std::ofstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return;
    }
    for (int i = 0;i<dataPointsLength;i++) {
        file << dataPoints[4*i] << "," << dataPoints[4*i+1] << "," << dataPoints[4*i+2] << "," << dataPoints[4*i+3] << "\n";
    }
    file.close();

}