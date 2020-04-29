#ifndef _CLUSTER_HPP_
#define _CLUSTER_HPP_

#include <iostream>
#include <vector>

class FaceClusterImp;
using namespace std;

class FaceCluster {
 public:
    FaceCluster(float thresh);
    virtual ~FaceCluster();
    std::pair<unsigned long, std::vector<unsigned long>> Cluster(std::vector<std::vector<float>>& descriptors);
    std::pair<unsigned long, std::vector<unsigned long>> Cluster(std::vector<std::vector<float>>& descriptors, std::pair<unsigned long, std::vector<unsigned long>>& labels);
    std::vector<float> Metric(void);
 private:
    FaceClusterImp* fc_imp_;
};

#endif /*_CLUSTER_HPP_*/

