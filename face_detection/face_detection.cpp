#include <string>
#include "face_detection.hpp"
#include "face_detection_imp.hpp"

using namespace std;

FaceDetection::FaceDetection(const string& prototxt,
                                      const string& model,
                                      vector<float>& mean,
                                      float thresh)
{

    fd_imp_ = new FaceDetectionImp(prototxt, model, mean, thresh);
}

FaceDetection::~FaceDetection()
{
    if (fd_imp_) {
        delete fd_imp_;
        fd_imp_ = nullptr;
    }
}

vector<vector<float>> FaceDetection::Detect(const cv::Mat & img)
{
    return fd_imp_->Detect(img);
}