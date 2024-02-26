p = [2.3,4.2,4,2,1,6.5,5,75.3,2.54,23.2]
centroids = [23.2,2.3]
import math
def edist(x1,x2):
    return math.sqrt((x2-x1)*(x2-x1))


clusterLabels = []
def clustAssing():
    for i in range(len(p)):
        minDist = math.inf
        clusterLabel = 0
        for j in range(len(centroids)):
            distance = edist(p[i],centroids[j])
            if distance < minDist:
                minDist = distance
                clusterLabel = j

        clusterLabels.append(clusterLabel);

clustAssing()
print(clusterLabels)


#index: 0 distance: 20.900002 j: 0  dataPoint: 2.300000 centroid: 23.200001 
print(edist(2.300000,23.200001 ))