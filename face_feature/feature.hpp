#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FeatureExtractionImp;

using namespace std;
class FeatureExtraction {
public:
    FeatureExtraction(const string& prototxt, const string& model,
        float mean, float scale, bool norm);
    virtual ~FeatureExtraction();
    std::vector<float> Extract(const cv::Mat& img);

private:
    FeatureExtractionImp* fe_imp_;
};

#endif /*_FEATURE_IMP_HPP_*/

