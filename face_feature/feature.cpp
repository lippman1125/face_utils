#include "feature.hpp"
#include "feature_imp.hpp"
#include <string>

using namespace std;

FeatureExtraction::FeatureExtraction(const string& prototxt,
                                                const string& model,
                                                float mean,
                                                float scale,
                                                bool norm)
{
    fe_imp_ = new FeatureExtractionImp(prototxt, model, mean, scale, norm);
}

FeatureExtraction::~FeatureExtraction()
{
    if (fe_imp_) {
        delete fe_imp_;
        fe_imp_ = nullptr;
    }
}

std::vector<float> FeatureExtraction::Extract(const cv::Mat& img)
{
    return fe_imp_->Extract(img);
}