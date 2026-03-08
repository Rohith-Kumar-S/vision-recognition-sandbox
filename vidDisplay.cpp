/*
  Rohith Kumar Senthil Kumar
  5330 Computer Vision

  Main file for Object Recognition project.
  Contains the main loop for video processing, user interaction, and integration of all components.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <queue>
#include <cstdlib>
#include "projectutils.h"
#include "csv_util/csv_util.h"
#include <opencv2/core/utils/logger.hpp>

int main(int argc, char *argv[])
{
    // main function for video processing and user interaction
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    char filename[256];
    bool isVideo = true;
    if (argc > 1)
    {
        strcpy(filename, argv[1]);
        if (ProjectUtils::isPng(filename))
            isVideo = false;
    }
    else
        strcpy(filename, "");

    // initialization of variables and objects
    cv::Mat frame, dst, controls, label, segmented;
    cv::Mat labels, stats, centroids, thresholded, embimage, embedding;
    std::vector<float> features;
    std::vector<cv::Point> corners;
    std::vector<float> parameters;
    cv::VideoCapture *capdev = nullptr;
    cv::dnn::Net net = cv::dnn::readNet("resnet18-v2-7.onnx");
    ProjectUtils utils;
    std::vector<char *> labelNames;
    std::vector<std::vector<float>> featureVectors;
    std::vector<float> stdev;
    std::vector<float> generatedFeatureVector;

    if (net.empty())
    {
        std::cerr << "Error: Could not load network." << std::endl;
        return -1;
    }

    // open video or image file
    if (isVideo)
    {

        if (strcmp(filename, "") == 0)
            // open phone camera if no video file is provided
            capdev = new cv::VideoCapture(1);

        else
            // open video file if provided
            capdev = new cv::VideoCapture(filename);

        if (!capdev->isOpened())
        {
            std::cout << "Unable to open video device\n";
            return -1;
        }

        cv::Size refS(
            capdev->get(cv::CAP_PROP_FRAME_WIDTH),
            capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

        printf("Expected size: %d %d\n", refS.width, refS.height);
    }
    else
    {
        frame = cv::imread(filename);
        std::cout << "frame shape: " << frame.cols << "x" << frame.rows << std::endl;
        if (frame.empty())
        {
            std::cout << "Unable to read image file\n";
            return -1;
        }
    }

    // Layout initialization
    cv::namedWindow("Display window", 1);
    cv::namedWindow("Controls", 1);
    cv::setMouseCallback("Controls", utils.onMouse, &utils);
    cv::Size nextTextSize = cv::getTextSize("next", cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
    cv::Size quitTextSize = cv::getTextSize("quit", cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
    cv::Size resnetTextSize = cv::getTextSize("resnet", cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
    cv::Size trainTextSize = cv::getTextSize("train", cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
    cv::Point nextTextPoint(
        utils.nextRect.x + (utils.nextRect.width - nextTextSize.width) / 2,
        utils.nextRect.y + (utils.nextRect.height + nextTextSize.height) / 2);
    cv::Point quitTextPoint(
        utils.quitRect.x + (utils.quitRect.width - quitTextSize.width) / 2,
        utils.quitRect.y + (utils.quitRect.height + quitTextSize.height) / 2);
    cv::Point resnetTextPoint(
        utils.resnetRect.x + (utils.resnetRect.width - resnetTextSize.width) / 3,
        utils.resnetRect.y + (utils.resnetRect.height + resnetTextSize.height) / 2);
    cv::Point trainTextPoint(
        utils.trainRect.x + (utils.trainRect.width - trainTextSize.width) / 3,
        utils.trainRect.y + (utils.trainRect.height + trainTextSize.height) / 2);

    while (true)
    {
        if (!utils.isDataloaded)
        {
            std::cout << "Loading data..." << std::endl;
            utils.loadData(labelNames, featureVectors, stdev);
            utils.isDataloaded = true;
        }

        if (isVideo)
        {
            *capdev >> frame;
        }
        if (frame.empty())
        {
            std::cout << "frame is empty\n";
            break;
        }
        if (frame.cols > 1000)
        {
            cv::resize(frame, frame,
                       cv::Size(frame.cols / 2, frame.rows / 2),
                       0, 0, cv::INTER_LINEAR);
        }
        dst = frame.clone();
        controls = cv::Mat::zeros(cv::Size(frame.cols + 100, 60), frame.type());
        // draw control buttons and layout
        utils.buildLayout(controls, nextTextPoint, quitTextPoint, resnetTextPoint, trainTextPoint, utils.isResnetModel, utils.isTrain);

        if (utils.isTrain)
            // run the training pipeline if in training mode, otherwise run inference pipeline
            utils.runTrainingPipeline(utils.mode, frame, dst, thresholded, labels, stats, centroids, corners, parameters, embedding, net, embimage, utils.featuresGenerated, utils.isFeatureStored, utils.segmented, utils.labelSubmitted, utils.isResnetModel, features, utils.labelId, utils.label);
        else
            // run inference pipeline
            utils.runInferencePipeline(frame, dst, segmented, thresholded, labels, stats, centroids, corners, parameters, embedding, net, embimage, featureVectors, labelNames, stdev, utils.isResnetModel, utils.segmented, generatedFeatureVector);

        // display the resulting frame and controls
        cv::imshow("Display window", dst);
        cv::imshow("Controls", controls);

        // Layout and user interaction handling
        int key = cv::waitKey(10) & 0xFF;
        if ((key == 'q' && !utils.recordLabel) || utils.quit)
        {
            break;
        }
        if ((utils.mode == "segmentation" && key == 'N') ||
            (utils.mode == "store-features" && key == 'N'))
        {
            utils.recordLabel = true;
            utils.label = ""; 
            cv::namedWindow("Labeling", 1);
            cv::setMouseCallback("Labeling", utils.onMouseLabel, &utils);
        }
        if (utils.recordLabel)
        {
            // delete last character allow backsapce
            if ((key == 8 || key == 127) && !utils.label.empty())
            {
                utils.label.pop_back();
            }
            // allow numeric input for segmentation labels and normal characters for feature storage labels
            else if (utils.mode == "segmentation" && key >= '0' && key <= '9')
            {
                utils.label += static_cast<char>(key);
            }
            // Normal printable characters for feature storage labels
            else if (utils.mode == "store-features" && key >= 32 && key <= 126)
            {
                utils.label += static_cast<char>(key);
            }
            label = cv::Mat::zeros(cv::Size(460, 60), frame.type());
            std::string descriptor =
                (utils.mode == "store-features") ? "Enter Label/Name: "
                                                 : "Enter Region ID: ";
            std::string labelDisplay = descriptor + utils.label;
            cv::putText(label, labelDisplay, cv::Point(5, 35), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            cv::rectangle(label, utils.submitRect, cv::Scalar(204, 204, 204), -1);
            cv::putText(label, "submit", cv::Point(utils.submitRect.x + 10, utils.submitRect.y + 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            cv::imshow("Labeling", label);
        }
    }
    cv::destroyAllWindows();
    return 0;
}