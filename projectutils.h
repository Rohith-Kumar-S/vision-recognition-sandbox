/*
  Rohith Kumar Senthil Kumar
  5330 Computer Vision

  Utility class for Object Recognition project.
*/

#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <unordered_map>
#include "utilities.cpp"
#include "csv_util/csv_util.h"

using std::string;

class ProjectUtils
{
public:
    // Interaction states
    cv::Rect nextRect{580, 10, 80, 40};
    cv::Rect submitRect{360, 10, 80, 40};
    cv::Rect quitRect{490, 10, 80, 40};
    cv::Rect resnetRect{360, 10, 120, 40};
    cv::Rect trainRect{260, 10, 90, 40};
    cv::Point photoCircle{700, 30};
    cv::Point modeTextCoordinates{20, 35};
    int labelId = 0;
    int circleRadius = 20;
    bool quit = false, capturePhoto = false, firstFrame = true, featuresGenerated = false, segmented = false, isFeatureStored = false, recordLabel = false, labelSubmitted = false, recordLabelId = false, labelIdSubmitted = false, isTrain = false, isResnetModel = false, isDataloaded = false;
    bool startCountdown = false;
    int countDown = 10;
    std::string mode = "threshold", label = "";

    void loadData(std::vector<char *> &labelNames, std::vector<std::vector<float>> &featureVectors, std::vector<float> &stdev)
    {
        // load features and labels from csv file, compute standard deviation
        // for each feature for normalization during inference
        if (!isResnetModel)
        {
            // load default features and labels from csv file
            featureVectors.clear();
            labelNames.clear();
            stdev.clear();
            read_image_data_csv("features.csv",
                                labelNames, featureVectors, 0);
            stdev = computeStdDeviation(featureVectors);
            std::cout << "Loaded default features" << std::endl;
        }
        else
        {
            // load resnet features and labels from csv file
            featureVectors.clear();
            labelNames.clear();
            stdev.clear();
            read_image_data_csv("resnet_features.csv",
                                labelNames, featureVectors, 0);
            std::cout << "Loaded resnet features" << std::endl;
        }
    }

    void buildLayout(cv::Mat &controls, cv::Point nextTextPoint, cv::Point quitTextPoint, cv::Point resnetTextPoint, cv::Point trainTextPoint, bool isResnetModel = false, bool isTrain = false)
    {
        // build the control window layout with buttons and text

        std::string resnetText = "resnet";
        std::string trainText = "train";
        if (isResnetModel)
            resnetText = "resnet on";
        else
            resnetText = "resnet off";
        if (isTrain)
            trainText = "train on";
        else
            trainText = "train off";

        // build the controls layout
        cv::rectangle(controls, nextRect, cv::Scalar(204, 204, 204), -1);
        cv::rectangle(controls, quitRect, cv::Scalar(204, 204, 204), -1);
        cv::rectangle(controls, resnetRect, cv::Scalar(204, 204, 204), -1);
        cv::rectangle(controls, trainRect, cv::Scalar(204, 204, 204), -1);
        cv::putText(controls, "next", nextTextPoint, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(controls, "quit", quitTextPoint, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(controls, resnetText, resnetTextPoint, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(controls, trainText, trainTextPoint, cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(controls, "mode: " + mode, modeTextCoordinates, cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
        cv::circle(controls, photoCircle, circleRadius, cv::Scalar(0, 0, 255), -1);
    }

    static void onMouse(int event, int x, int y, int flags, void *userdata)
    {
        // onMouse: handle mouse click events for the control window

        ProjectUtils *self = static_cast<ProjectUtils *>(userdata);

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            if (self->nextRect.contains(cv::Point(x, y)))
            {
                // toggle through modes on next button click
                std::cout << "Next clicked!" << std::endl;
                if (self->mode == "threshold")
                    self->mode = "segmentation";
                else if (self->mode == "segmentation")
                    self->mode = "extract-features";
                else if (self->mode == "extract-features")
                    self->mode = "store-features";
                else
                    self->mode = "threshold";
            }
            else if (self->quitRect.contains(cv::Point(x, y)))
            {
                // quit the application on quit button click
                std::cout << "Quitting!" << std::endl;
                self->quit = true;
            }
            else if (self->resnetRect.contains(cv::Point(x, y)))
            {
                // toggle resnet model on/off on resnet button click
                std::cout << "Resnet toggle!" << std::endl;
                self->isResnetModel = !self->isResnetModel;
                if (!self->isTrain)
                {
                    self->isDataloaded = false;
                }
            }
            else if (self->trainRect.contains(cv::Point(x, y)))
            {
                // toggle train mode on/off on train button click
                std::cout << "Train toggle!" << std::endl;
                self->isTrain = !self->isTrain;
                if (!self->isTrain)
                {
                    self->isDataloaded = false;
                }
            }
            else if (self->submitRect.contains(cv::Point(x, y)))
            {
                // submit the label on submit button click
                std::cout << "Submitting label!" << std::endl;
                self->recordLabel = false;
                cv::destroyWindow("Labeling");
            }
        }
    }

    static void onMouseLabel(int event, int x, int y, int flags, void *userdata)
    {

        // onMouseLabel: handle mouse click events for the labeling window
        ProjectUtils *self = static_cast<ProjectUtils *>(userdata);

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            if (self->submitRect.contains(cv::Point(x, y)))
            {
                std::cout << "Submitting label!" << self->label << std::endl;
                self->recordLabel = false;
                self->labelSubmitted = true;
                if (self->mode == "segmentation")
                    self->labelId = std::stoi(self->label);
                cv::destroyWindow("Labeling");
            }
        }
    }

    static std::vector<float> computeStdDeviation(const std::vector<std::vector<float>> &featureVectors)
    {

        // compute standard deviation for each feature across all feature vectors for normalization during inference
        int n = featureVectors.size();

        int D = featureVectors[0].size();

        std::vector<float> mean(D, 0.0f);
        std::vector<float> stdev(D, 0.0f);

        // compute mean for each feature
        for (int i = 0; i < D; i++)
        {
            for (int j = 0; j < n; j++)
            {
                mean[i] += featureVectors[j][i];
            }
            mean[i] /= n;
        }

        // compute standard deviation for each feature
        for (int i = 0; i < D; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float centered = featureVectors[j][i] - mean[i];
                stdev[i] += centered * centered;
            }
            stdev[i] = std::sqrt(stdev[i] / n);

            //  For Numerical stability
            if (stdev[i] < 1e-6f)
                stdev[i] = 1e-6f;
        }

        return stdev;
    }

    static std::string findNearestNeighbor(const std::vector<float> &featureVector, const std::vector<std::vector<float>> &featureVectors, const std::vector<float> &stdev, const std::vector<char *> &labelNames, bool isResnetModel = false)
    {
        // find the nearest neighbor in the feature space and return the
        // corresponding label, using either Euclidean distance or cosine similarity based on the model type

        int n = featureVectors.size();
        int D = featureVector.size();
        std::priority_queue<std::pair<float, std::string>,
                            std::vector<std::pair<float, std::string>>,
                            std::greater<std::pair<float, std::string>>>
            minHeap;

        std::vector<float> distances(n, 0.0f);
        for (int i = 0; i < n; i++)
        {
            float dist = 0.0f;
            if (isResnetModel)
            {
                float dot = 0.0f, normA = 0.0f, normB = 0.0f;
                for (int j = 0; j < D; j++)
                {
                    dot += featureVector[j] * featureVectors[i][j];
                    normA += featureVector[j] * featureVector[j];
                    normB += featureVectors[i][j] * featureVectors[i][j];
                }
                dist = 1.0f - (dot / (std::sqrt(normA) * std::sqrt(normB)));
            }
            else
            {
                for (int j = 0; j < D; j++)
                {
                    float diff = (featureVector[j] - featureVectors[i][j]) / stdev[j];
                    dist += (diff * diff);
                }
                dist = std::sqrt(dist);
            }
            minHeap.push({dist, labelNames[i]});
        }

        std::string resultLabel = "Unknown";

        // Threshold for classification
        if (minHeap.top().first < (isResnetModel ? 0.25f : 2.0f))
        {
            std::cout << "Predicted Label: " << minHeap.top().second << " | Distance: " << minHeap.top().first << std::endl;
            resultLabel = minHeap.top().second;
        }
        else
        {
            std::cout << "Unknown. Closest Label: " << minHeap.top().second << " | Distance: " << minHeap.top().first << std::endl;
        }
        return resultLabel;
    }

    static void threshold(cv::Mat &src, cv::Mat &dst, double thresValue)
    {
        // apply Gaussian blur to reduce noise, convert to grayscale,
        // apply binary thresholding, and then morphological closing to fill gaps in the segmented objects
        cv::Mat srcBlurred, srcGray, thresholded;

        // apply Gaussian blur to reduce noise before thresholding
        cv::GaussianBlur(src, srcBlurred, cv::Size(5, 5), 1.5);
        cv::cvtColor(srcBlurred, srcGray, cv::COLOR_BGR2GRAY);
        // Create output image
        thresholded = cv::Mat::zeros(srcGray.size(), srcGray.type());

        // Apply THRESH_BINARY_INV to invert the thresholding result, so that objects are white and background is black
        for (int y = 0; y < srcGray.rows; y++)
        {
            for (int x = 0; x < srcGray.cols; x++)
            {
                if (static_cast<double>(srcGray.at<uchar>(y, x)) > thresValue)
                    thresholded.at<uchar>(y, x) = 0;
                else
                    thresholded.at<uchar>(y, x) = 255;
            }
        }

        // apply morphological closing to fill gaps in the segmented objects
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(thresholded, dst, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    }

    static bool isPng(const char *filename)
    {
        // check if the file is a PNG based on its extension
        if (!filename)
            return false;

        size_t len = std::strlen(filename);
        if (len < 4)
            return false;

        return std::tolower(filename[len - 4]) == '.' &&
               std::tolower(filename[len - 3]) == 'p' &&
               std::tolower(filename[len - 2]) == 'n' &&
               std::tolower(filename[len - 1]) == 'g';
    }

    static int segmentImage(cv::Mat &src, cv::Mat &binaryImage, cv::Mat &dst, cv::Mat labels, cv::Mat stats, cv::Mat centroids, int numLabels)
    {
        // segment the image using connected components analysis,
        // filter out small noise components based on area,
        // relabel the remaining components, and visualize the results with different colors and centroids

        float colors[5][3] = {
            {218, 99, 93},
            {246, 194, 107},
            {126, 185, 99},
            {73, 161, 197},
            {164, 97, 189}};

        dst = cv::Mat::zeros(src.size(), CV_8UC3);

        // Filter out small components based on area and create a mapping from old labels to new labels
        int minArea = 1000;
        int validRegions = 0;
        std::unordered_map<int, int> newLabelMapping;
        std::unordered_map<int, int> reverseMap;

        // Remap labels based on area and count valid regions
        for (int i = 0; i < numLabels; i++)
        {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < minArea)
                continue;
            newLabelMapping[i] = validRegions;
            validRegions++;
        }

        // Create reverse mapping for centroids access
        for (auto &[oldLabel, newLabel] : newLabelMapping)
        {
            reverseMap[newLabel] = oldLabel;
        }

        // Relabel the components in the labels matrix based on the new mapping
        for (int y = 0; y < labels.rows; y++)
        {
            for (int x = 0; x < labels.cols; x++)
            {
                auto it = newLabelMapping.find(labels.at<int>(y, x));
                labels.at<int>(y, x) = (it != newLabelMapping.end()) ? it->second : 0;
            }
        }

        // Apply the segmented components with different colors and centroids
        for (int y = 0; y < dst.rows; y++)
        {
            for (int x = 0; x < dst.cols; x++)
            {
                int label = labels.at<int>(y, x);
                if (label != 0)
                {
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(colors[label % 5][0], colors[label % 5][1], colors[label % 5][2]);
                }
            }
        }

        // Draw centroids and label them with their new labels
        for (int i = 0; i < validRegions; i++)
        {
            int cx = static_cast<int>(centroids.at<double>(reverseMap[i], 0));
            int cy = static_cast<int>(centroids.at<double>(reverseMap[i], 1));
            cv::putText(dst, std::to_string(i), cv::Point(cx + 10, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
        return validRegions;
    }

    static void drawBoundingBoxes(cv::Mat &src, cv::Mat &dst, std::vector<cv::Point> &corners, std::vector<float> parameters)
    {
        // draw bounding boxes around the segmented objects based on the computed corners and orientation parameters

        float maxX = parameters[0];
        float minX = parameters[1];
        float maxY = parameters[2];
        float minY = parameters[3];
        float xc = parameters[4];
        float yc = parameters[5];
        float c = parameters[6];
        float s = parameters[7];

        double axisLength = (maxX - minX) / 2.0;

        // Axis endpoints in image space
        cv::Point center(static_cast<int>(xc), static_cast<int>(yc));
        cv::Point axisEnd(
            static_cast<int>(xc + axisLength * c),
            static_cast<int>(yc + axisLength * s));

        // Draw the axis
        cv::line(dst, center, axisEnd, cv::Scalar(255), 2);

        for (size_t i = 0; i < corners.size(); i++)
        {
            cv::line(dst, corners[i], corners[(i + 1) % corners.size()], cv::Scalar(0, 255, 0), 2);
        }
    }

    static void generateFeatures(cv::Mat &binaryImage, std::vector<float> &features, cv::Mat labels, std::vector<cv::Point> &corners, std::vector<float> &parameters, int labelToAnalyze = 6)
    {
        // generate features for a specific segmented object based on its label, including area, centroid, orientation, and bounding box parameters

        int minArea = 1000;

        double m10 = 0.0, m01 = 0.0, m00 = 0.0, xc = 0.0, yc = 0.0, m11 = 0.0,
               m20 = 0.0, m02 = 0.0, mu11 = 0.0, mu20 = 0.0, mu02 = 0.0, angle = 0.0;

        // Compute spatial moments for the specified label
        for (int y = 0; y < binaryImage.rows; y++)
        {
            for (int x = 0; x < binaryImage.cols; x++)
            {
                if (labels.at<int>(y, x) == labelToAnalyze && binaryImage.at<uchar>(y, x) > 0)
                {
                    m10 += x;
                    m01 += y;
                    m00 += 1.0;
                    m11 += x * y;
                    m20 += x * x;
                    m02 += y * y;
                }
            }
        }

        if (m00 > 0)
        {
            xc = m10 / m00;
            yc = m01 / m00;
        }

        // Compute central moments for orientation calculation
        for (int y = 0; y < binaryImage.rows; y++)
        {
            for (int x = 0; x < binaryImage.cols; x++)
            {
                if (labels.at<int>(y, x) == labelToAnalyze && binaryImage.at<uchar>(y, x) > 0)
                {
                    mu11 += (x - xc) * (y - yc);
                    mu02 += (y - yc) * (y - yc);
                    mu20 += (x - xc) * (x - xc);
                }
            }
        }

        // Calculate orientation angle using central moments
        angle = 0.5 * std::atan2(2 * mu11, mu20 - mu02);

        // Rotate the coordinates of the pixels belonging to the object to find the bounding box in the rotated coordinate system
        double c = std::cos(angle);
        double s = std::sin(angle);
        double minX = std::numeric_limits<double>::max();
        double maxX = -std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double maxY = -std::numeric_limits<double>::max();

        for (int y = 0; y < binaryImage.rows; y++)
        {
            for (int x = 0; x < binaryImage.cols; x++)
            {
                if (labels.at<int>(y, x) == labelToAnalyze && binaryImage.at<uchar>(y, x) > 0)
                {
                    double xRot = c * (x - xc) + s * (y - yc);
                    double yRot = -s * (x - xc) + c * (y - yc);
                    minX = std::min(minX, xRot);
                    maxX = std::max(maxX, xRot);
                    minY = std::min(minY, yRot);
                    maxY = std::max(maxY, yRot);
                }
            }
        }

        std::vector<std::pair<double, double>> roiCorners = {
            {minX, minY}, {maxX, minY}, {maxX, maxY}, {minX, maxY}};

        for (auto &[rx, ry] : roiCorners)
        {
            // Rotate the corners back to image space
            double ix = c * rx - s * ry + xc;
            double iy = s * rx + c * ry + yc;
            corners.push_back(cv::Point(static_cast<int>(ix), static_cast<int>(iy)));
        }

        double axisLength = (maxX - minX) / 2.0;

        // Axis endpoints in image space
        cv::Point center(static_cast<int>(xc), static_cast<int>(yc));
        cv::Point axisEnd(
            static_cast<int>(xc + axisLength * c),
            static_cast<int>(yc + axisLength * s));

        // Calculate width and height of the bounding box
        float w = std::sqrt(
            (corners[0].x - corners[1].x) * (corners[0].x - corners[1].x) +
            (corners[0].y - corners[1].y) * (corners[0].y - corners[1].y));

        // Height
        float h = std::sqrt(
            (corners[0].x - corners[3].x) * (corners[0].x - corners[3].x) +
            (corners[0].y - corners[3].y) * (corners[0].y - corners[3].y));

        // Percent filled
        float centFilled = m00 / (w * h);

        cv::Mat mask = (labels == labelToAnalyze); // CV_8U mask
        cv::Moments m = cv::moments(mask, true);

        // Calculate Hu Moments
        double hu[7];
        cv::HuMoments(m, hu);

        float aspectRatio = h / w;

        // Append features to the feature vector
        features.push_back(centFilled);
        features.push_back(aspectRatio);
        for (int i = 0; i < 7; i++)
            features.push_back(-std::copysign(1.0, hu[i]) * std::log10(std::abs(hu[i])));

        // Append parameters for visualization and debugging
        parameters.push_back(maxX);
        parameters.push_back(minX);
        parameters.push_back(maxY);
        parameters.push_back(minY);
        parameters.push_back(xc);
        parameters.push_back(yc);
        parameters.push_back(c);
        parameters.push_back(s);
        parameters.push_back(angle);
    }

    static void runTrainingPipeline(std::string mode, cv::Mat &frame, cv::Mat &dst, cv::Mat &thresholded, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids, std::vector<cv::Point> &corners, std::vector<float> &parameters, cv::Mat &embedding, cv::dnn::Net &net, cv::Mat &embimage, bool &featuresGenerated, bool &isFeatureStored, bool &segmented, bool &labelSubmitted, bool &isResnetModel, std::vector<float> &features, int &labelId, std::string &label)
    {
        // run the training pipeline based on the current mode,
        // including thresholding, segmentation, feature extraction, and feature storage with label submission

        if (mode == "threshold")
        {
            threshold(frame, dst, 120.0);
            featuresGenerated = false;
            isFeatureStored = false;
            segmented = false;
            labelSubmitted = false;
        }
        else if (mode == "segmentation")
        {
            int numLabels = cv::connectedComponentsWithStats(
                thresholded, labels, stats, centroids, 8);

            ProjectUtils::segmentImage(frame, thresholded, dst, labels, stats, centroids, numLabels);
        }
        else if (mode == "extract-features")
        {

            labelSubmitted = false;
            if (!featuresGenerated)
            {
                corners.clear();
                parameters.clear();
                features.clear();
                ProjectUtils::generateFeatures(thresholded, features, labels, corners, parameters, labelId);
                if (isResnetModel)
                {
                    prepEmbeddingImage(frame, embimage,
                                       (int)parameters[4], (int)parameters[5], // cx, cy
                                       parameters[8],                          // theta
                                       parameters[1], parameters[0],           // minE1, maxE1
                                       -parameters[2], -parameters[3],         // minE2, maxE2 (swapped!)
                                       1);

                    getEmbedding(embimage, embedding, net, 0);
                }
            }
            ProjectUtils::drawBoundingBoxes(frame, dst, corners, parameters);
            featuresGenerated = true;
        }
        else if (mode == "store-features")
        {
            ProjectUtils::drawBoundingBoxes(frame, dst, corners, parameters);
            if (!isFeatureStored && labelSubmitted)
            {
                if (isResnetModel)
                {
                    features.clear();
                    features.assign(embedding.begin<float>(), embedding.end<float>());
                    append_image_data_csv("resnet_features.csv", const_cast<char *>(label.c_str()), features, 0);
                }
                else
                {
                    append_image_data_csv("features.csv", const_cast<char *>(label.c_str()), features, 0);
                }
                isFeatureStored = true;
            }
        }
    }

    static void runInferencePipeline(cv::Mat &frame, cv::Mat &dst, cv::Mat &segmented, cv::Mat &thresholded, cv::Mat &labels, cv::Mat &stats, cv::Mat &centroids, std::vector<cv::Point> &corners, std::vector<float> &parameters, cv::Mat &embedding, cv::dnn::Net &net, cv::Mat &embimage, std::vector<std::vector<float>> &featureVectors, std::vector<char *> &labelNames, std::vector<float> &stdev, bool isResnetModel, bool isSegmented, std::vector<float> &generatedFeatureVector)
    {
        // run the inference pipeline, including thresholding, 
        // optional segmentation, feature extraction, and nearest neighbor classification with visualization of results
        
        threshold(frame, thresholded, 120.0);
        int numLabels = 0;
        if (!isSegmented)
            numLabels = cv::connectedComponentsWithStats(
                thresholded, labels, stats, centroids, 8);
        int validRegions = ProjectUtils::segmentImage(frame, thresholded, segmented, labels, stats, centroids, numLabels);
        for (int i = 1; i < validRegions; i++)
        {
            generatedFeatureVector.clear();
            corners.clear();
            parameters.clear();
            ProjectUtils::generateFeatures(thresholded, generatedFeatureVector, labels, corners, parameters, i);

            if (isResnetModel)
            {
                embimage.release();
                embedding.release();
                generatedFeatureVector.clear();

                prepEmbeddingImage(frame, embimage,
                                   (int)parameters[4], (int)parameters[5],
                                   parameters[8],
                                   parameters[1], parameters[0],
                                   -parameters[2], -parameters[3],
                                   0);
                getEmbedding(embimage, embedding, net, 0);
                generatedFeatureVector.assign(embedding.begin<float>(), embedding.end<float>());
            }

            std::string predictedLabel = findNearestNeighbor(generatedFeatureVector, featureVectors, stdev, labelNames, isResnetModel);
            if (predictedLabel != "Unknown")
            {
                ProjectUtils::drawBoundingBoxes(frame, dst, corners, parameters);
                cv::putText(dst, predictedLabel, corners[1] + cv::Point(20, 5), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }
    }
};