#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include "face_detection_imp.hpp"

using namespace caffe;
using namespace std;

FaceDetectionImp::FaceDetectionImp(const string& prototxt,
                                             const string& model,
                                             vector<float>& mean,
                                             float thresh) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    net_.reset(new Net<float>(prototxt, TEST));
    net_->CopyTrainedLayersFrom(model);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    std::vector<cv::Mat> channels;
    for (unsigned int i = 0; i < mean.size(); ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1, cv::Scalar(mean[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
    thresh_ = thresh;

}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void FaceDetectionImp::WrapInputLayer(vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void FaceDetectionImp::Preprocess(const cv::Mat& img, vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    sample_normalized = sample_float - mean_;

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

vector<vector<float>> FaceDetectionImp::PostProcess(const cv::Mat& img) {
    /* Copy the output layer to a std::vector */
    Blob<float>* result_blob = net_->output_blobs()[0];
    const float* result = result_blob->cpu_data();
    int num_det = result_blob->height();
    cout<<"detection num = "<< num_det <<endl;
    // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
    vector<vector<float> > detections;
    for (int k = 0; k < num_det; ++k) {
        if (result[0] == -1 || result[2] < thresh_) {
          // Skip invalid detection.
          result += 7;
          continue;
        }
        float x1 = result[3] * img.cols;
        float y1 = result[4] * img.rows;
        float x2 = result[5] * img.cols;
        float y2 = result[6] * img.rows;
        vector<float> detection(result, result+7);
        detection[3] = x1 < 0 ? 0 : x1;
        detection[4] = y1 < 0 ? 0 : y1;
        detection[5] = x2 > img.cols - 1 ? img.cols - 1 : x2;
        detection[6] = y2 > img.rows - 1 ? img.rows - 1 : y2;
        detections.push_back(detection);
        result += 7;
    }
    return detections;
}

vector<vector<float>> FaceDetectionImp::Detect(const cv::Mat& img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    net_->Forward();
    return PostProcess(img);
}
