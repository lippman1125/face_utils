#ifndef _FACEDECTION_HPP_
#define _FACEDECTION_HPP_

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

class FaceDetectionImp;
class FaceDetection{
 public:
    FaceDetection(const string& prototxt, const string& model, vector<float>& mean, float thresh);
    virtual ~FaceDetection();
    vector<vector<float>> Detect(const cv::Mat& img);

 private:
    FaceDetectionImp* fd_imp_;
};

#endif /*_FACEDECTION_HPP_*/

