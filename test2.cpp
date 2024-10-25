#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <dirent.h>

using namespace std;

const int HIST_SIZE = 16;

// Function to compute sum of squared differences between two feature vectors
float SSD(const vector<float>& feature1, const vector<float>& feature2) {
    float ssd = 0;
    for (size_t i = 0; i < feature1.size(); ++i) {
        float diff = feature1[i] - feature2[i];
        ssd += diff * diff;
    }
    return ssd;
}

float histIntersection(const vector<float>& feature1, const vector<float>& feature2) {
    float histDist = 0;

    for (size_t i = 0; i < feature1.size(); ++i) {
        histDist += min(feature1[i], feature2[i]); // Sum of minimum values across all bins
    }

    return 1-histDist;
}

// Extracts 7x7 middle of the image as feature vector
vector<float> baseline(cv::Mat &image) {
    int x = (image.cols - 3) / 2;
    int y = (image.rows - 3) / 2;

    cv::Rect roi(x, y, 7, 7);
    cv::Mat featureRegion = image(roi);

    vector<float> featureVector;
    for (int y = 0; y < featureRegion.rows; ++y) {
        for (int x = 0; x < featureRegion.cols; ++x) {
            featureVector.push_back(featureRegion.at<uchar>(y, x));
        }
    }

    return featureVector;
}

vector<float> computeHistogram(cv::Mat src) {
    vector<float> hist(HIST_SIZE * HIST_SIZE, 0.0);

    for (int i = 0; i < src.rows; ++i) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);

        for (int j = 0; j < src.cols; ++j) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // Check for all zeros

            float r = R / divisor;
            float g = G / divisor;

            int rindex = (int)(r * (HIST_SIZE - 1) + 0.5);
            int gindex = (int)(g * (HIST_SIZE - 1) + 0.5);

            hist[rindex * HIST_SIZE + gindex]++;
        }
    }

    // Normalize the histogram
    float totalPixels = src.rows * src.cols;
    for (size_t i = 0; i < hist.size(); ++i) {
        hist[i] /= totalPixels;
    }

    return hist;
}

vector<float> singleHM(cv::Mat &src) {
    return computeHistogram(src);
}

vector<float> multiHM(cv::Mat &src) {
    vector<float> hist;

    // First histogram (whole image)
    hist = computeHistogram(src);

    // Second histogram (center 7x7)
    int x = (src.cols - 7) / 2;
    int y = (src.rows - 7) / 2;

    cv::Rect roi(x, y, 7, 7);
    cv::Mat centerRegion = src(roi);
    vector<float> centerHist = computeHistogram(centerRegion);

    // Concatenate histograms
    hist.insert(hist.end(), centerHist.begin(), centerHist.end());

    return hist;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        cout << "Error: Insufficient arguments." << endl;
        cout << "Usage: " << argv[0] << " <target image> <directory> <feature type> <output images>" << endl;
        return -1;
    }

    char targetImage[256];
    char dirname[256];
    strcpy(targetImage, argv[1]);
    strcpy(dirname, argv[2]);

    int featureType = atoi(argv[3]);
    int outputImages = atoi(argv[4]);

    DIR *dirp = opendir(dirname);
    if (dirp == NULL) {
        cout << "Cannot open directory: " << dirname << endl;
        return -1;
    }

    cv::Mat tImage = cv::imread(targetImage);
    if (tImage.empty()) {
        cout << "Error: Unable to read image: " << targetImage << endl;
        return -1;
    }

    vector<float> targetFeatures;
    switch (featureType) {
        case 1:
            targetFeatures = baseline(tImage);
            break;
        case 2:
            targetFeatures = singleHM(tImage);
            break;
        case 3:
            targetFeatures = multiHM(tImage);
            break;
        default:
            cout << "Invalid feature type." << endl;
            return -1;
    }

    vector<pair<float, string>> distances;

    struct dirent *dp;
    char buffer[256];

    while ((dp = readdir(dirp)) != NULL) {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            cv::Mat img = cv::imread(buffer);
            float distance;

            if (!img.empty()) {
                vector<float> imageFeatures;
                switch (featureType) {
                    case 1:
                        imageFeatures = baseline(img);
                        distance = SSD(targetFeatures, imageFeatures);
                        break;
                    case 2:
                        imageFeatures = singleHM(img);
                        distance = histIntersection(targetFeatures, imageFeatures);
                        // distance = SSD(targetFeatures, imageFeatures);
                        break;
                    case 3:
                        imageFeatures = multiHM(img);
                        distance = histIntersection(targetFeatures, imageFeatures);
                        break;
                }

                
                

                distances.push_back(make_pair(distance, dp->d_name));
            }
        }
    }
    closedir(dirp);

    sort(distances.begin(), distances.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    cv::Mat src;
    char buf[256];
    cout << "Top " << outputImages << " matches:" << endl;
    for (int i = 0; i < min(outputImages, static_cast<int>(distances.size())); ++i) {
        cout << "Distance: " << distances[i].first << ", Image: " << distances[i].second << endl;

        strcpy(buf, dirname);
        strcat(buf, "/");
        strcat(buf, distances[i].second.c_str());

        src = cv::imread(buf);
        if (!src.empty()) {
            cv::imshow(distances[i].second, src);
            cv::waitKey(0);
            cv::destroyWindow(distances[i].second);
        }
    }

    return 0;
}
