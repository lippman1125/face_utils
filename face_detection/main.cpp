#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>

#include "face_detection.hpp"

using namespace std;
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout <<"please input file"<<std::endl;
        return 0;
    }
    string path = string(argv[1]);
    int pos = path.find_last_of('/');
    string basename(path.substr(pos+1));
    pos = basename.find_last_of('.');
    string prefix(path.substr(0, pos));

    const string model_file = "yufacedetectnet-open-v1.prototxt";
    const string weights_file = "yufacedetectnet-open-v1.caffemodel";
    const string out_file = prefix + "_out.txt";
    const float conf_thresh = 0.8;
    vector<float> mean;
    mean.push_back(103.94);
    mean.push_back(116.78);
    mean.push_back(123.68);

    // Initialize the network.
    FaceDetection detector(model_file, weights_file, mean, conf_thresh);

    // Set the output mode.
    std::streambuf* buf = std::cout.rdbuf();
    std::ofstream outfile;
    if (!out_file.empty()) {
        outfile.open(out_file.c_str());
        if (outfile.good()) {
            buf = outfile.rdbuf();
        }
    }
    std::ostream out(buf);
    ostringstream ss;

    // Process image one by one.
    std::ifstream infile(argv[1]);
    std::cout<<"start reading files"<<std::endl;
    std::string file = argv[1];
    std::cout<<"reading filename succeed"<<std::endl;
    cv::Mat img = cv::imread(file, -1);

    CHECK(!img.empty()) << "Unable to decode image " << file;
    vector<vector<float>> detections = detector.Detect(img);
    /* Print the detection results. */
    for (unsigned int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        CHECK_EQ(d.size(), 7);
        const float score = d[2];
        out << prefix << " ";
        out << static_cast<int>(d[1]) << " ";
        out << score << " ";
        out << static_cast<int>(d[3]) << " ";
        out << static_cast<int>(d[4]) << " ";
        out << static_cast<int>(d[5]) << " ";
        out << static_cast<int>(d[6]) << std::endl;
        cv::Rect box(static_cast<int>(d[3]), static_cast<int>(d[4]),
            (static_cast<int>(d[5]) - static_cast<int>(d[3])),
            (static_cast<int>(d[6]) - static_cast<int>(d[4])));
        cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
        ss.str("");
        ss << score;
        cv::String conf(ss.str());
        cv::String label = static_cast<int>(d[1]) + ": " + conf;
        putText(img, label, cv::Point(static_cast<int>(d[3]), static_cast<int>(d[4])),
              0, 0.5, cv::Scalar(0,0,255));

    }
    cv::imwrite(prefix + "_out.jpg", img);
    // cv::waitKey();

    return 0;
}


