#include <fstream>
#include <iostream>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ctime>
#include <cmath>
#include <dirent.h>
#include <string.h>
#include <strings.h>
#include <iomanip>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "feature.hpp"


using namespace std;
int main(int argc, char * * argv) {
    if (argc != 2) {
        cout <<"face_feature face_image" << endl;
        return 0;
    }
    string prototxt = "/home/lqy/workshop/image_utils/face_recognition/mnasnet/mnas0.5-long-softmax-retina.prototxt";
    string model = "/home/lqy/workshop/image_utils/face_recognition/mnasnet/mnas0.5-long-combine_merge_retina-153-lfw0.996233.caffemodel";
    class FeatureExtraction feature_extraction(prototxt, model, 128, 128, true);
    vector<float> descriptor;

    cv::Mat img = cv::imread(argv[1]);
    cout.width(7);
    if (img.data != nullptr) {
        descriptor = feature_extraction.Extract(img);
        for (size_t i = 0; i < descriptor.size(); i++) {
            cout<<setiosflags(ios::fixed)<<setprecision(6)<<descriptor[i]<<endl;
        }
    }
    return 0;
}

