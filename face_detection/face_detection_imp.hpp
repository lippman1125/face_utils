#ifndef _FACEDECTION_IMP_HPP_
#define _FACEDECTION_IMP_HPP_

#include <caffe/caffe.hpp>
#include <memory>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;

class FaceDetectionImp {
 public:
    FaceDetectionImp(const string& prototxt, const string& model, vector<float>& mean, float thresh);
    vector<vector<float>> Detect(const cv::Mat& img);

 private:
    void WrapInputLayer(vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
    vector<vector<float>> PostProcess(const cv::Mat& img);

 private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    float thresh_;
};

#endif /*_FACEDECTION_IMP_HPP_*/
