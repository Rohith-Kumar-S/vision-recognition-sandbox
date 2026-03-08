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

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    char filename[256];

    // error checking
    if (argc < 2)
    {
        printf("usage: %s <image filename>\n", argv[0]);
        return (-1);
    }

    strcpy(filename, argv[1]);

    // read the file
    cv::Mat src, thresholded, dst;
    std::vector<float> features;
    std::vector<cv::Point> corners;
    std::vector<float> parameters;
    src = cv::imread(filename);
    if (src.data == NULL)
    {
        printf("error: unable to read filename %s\n", filename);
        return (-2);
    }
    cv::Mat labels, stats, centroids;
    
    ProjectUtils::threshold(src, thresholded, 100.0);
    int numLabels = cv::connectedComponentsWithStats(
        thresholded, labels, stats, centroids, 8);
    ProjectUtils::generateFeatures(thresholded, features, labels, corners, parameters, 6);
    // ProjectUtils::drawBoundingBoxes(src, dst, corners, parameters);
    ProjectUtils::segmentImage(src, thresholded, dst, labels, stats, centroids, numLabels);
    append_image_data_csv("features.csv", filename, features, 0);
    
    while (true)
    {

        cv::imshow("Display window", dst);
        if ((cv::waitKey(25) & 0xFF) == 'q')
            break;
    }
    cv::destroyAllWindows();
    return 0;
}