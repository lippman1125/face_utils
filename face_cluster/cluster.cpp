#include "cluster.hpp"
#include "cluster_imp.hpp"

FaceCluster::FaceCluster(float thresh)
{
    fc_imp_ = new FaceClusterImp(thresh);
}

FaceCluster::~FaceCluster()
{
    if (fc_imp_) {
        delete fc_imp_;
        fc_imp_= nullptr;
    }
}

std::pair<unsigned long, std::vector<unsigned long>>  FaceCluster::Cluster(vector<std::vector<float>> & descriptors)
{
    return fc_imp_->Cluster(descriptors);
}

std::pair<unsigned long, std::vector<unsigned long>>  FaceCluster::Cluster(vector<std::vector<float>> & descriptors, std::pair <unsigned long, std::vector<unsigned long>> & labels)
{
    return fc_imp_->Cluster(descriptors, labels);
}

std::vector<float> FaceCluster::Metric(void)
{
    return fc_imp_->Metric();
}