/*
    Different features and distance metrics implementation
*/

/*
    Difference between computeHistogram (Task 2,3,4) and colorHistogram (Task 7)

    1. computeHistogram:This function computes a histogram for an input image. It generates a histogram based on the intensity values of the pixels in the image. 
        This means it considers all color channels (usually three channels for RGB images) and computes a single histogram representing the distribution of intensity values across the entire image.

    2. computeColorHistogram: This function specifically computes a color histogram for an input image. 
        Unlike computeHistogram, which considers intensity values across all color channels, computeColorHistogram separates the image into its color channels (usually Red, Green, and Blue for RGB images) and computes a histogram for each channel individually. 
        This means it generates separate histograms representing the distribution of each color channel's intensity values.

        In summary, the main difference lies in the type of information captured by each histogram:

            computeHistogram captures overall intensity distribution of the image.
            computeColorHistogram captures individual color channel intensity distributions.

*/

#include<iostream> // Standard input/output stream
#include<vector>    // Standard vector container
#include<opencv2/opencv.hpp> // OpenCV library
#include<algorithm> // Standard algorithms library

using namespace std;


// Function to compute sum of squared differences between two feature vectors
float SSD(const vector<float>& feature1, const vector<float>& feature2) {
    float ssd = 0;
    for (size_t i = 0; i < feature1.size(); ++i) {
        ssd += pow(feature1[i] - feature2[i], 2); // Sum of squared differences calculation
    }
    return ssd;
}


// Function to compute histogram intersection between two feature vectors
float histIntersection(const vector<float>& feature1, const vector<float>& feature2){

    float histDist=0;

    for(size_t i=0;i<feature1.size();++i){
        histDist += std::min(feature1[i], feature2[i]); // Histogram intersection calculation
    }

    return 1-histDist;
}

// Function to compute cosine distance between two feature vectors
float cosineDistance(const vector<float>& feature1, const vector<float>& feature2){

    float cDist = 0;
    float dotProduct = 0;
    float norm1 = 0;
    float norm2 = 0;

    // Dot Product and Norm calculation
    for(size_t i=0;i<feature1.size();++i){
        dotProduct += feature1[i] * feature2[i]; // Dot Product
        norm1 += feature1[i] * feature1[i]; // Norm of feature vector 1
        norm2 += feature2[i] * feature2[i]; // Norm of feature vector 2
    }

    // Cosine distance calculation
    cDist = dotProduct / (sqrt(norm1) * sqrt(norm2));
    
    return 1 - cDist; // Return cosine distance
}

// Function to extract a baseline feature vector from the middle 7x7 region of an image
vector<float> baseline(cv::Mat &image){

    int x = (image.cols -3) / 2; // Calculate x-coordinate of the starting point for the 7x7 region
    int y = (image.rows -3) / 2; // Calculate y-coordinate of the starting point for the 7x7 region

    cv::Rect roi(x, y, 7,7); // Define a region of interest (ROI) for the middle 7x7 region
    cv::Mat featureRegion = image(roi); // Extract the middle 7x7 region from the image

    vector<float> featureVector; // Initialize feature vector
    for (int y = 0; y < featureRegion.rows; ++y) {
        for (int x = 0; x < featureRegion.cols; ++x) {
            featureVector.push_back(featureRegion.at<uchar>(y, x)); // Store pixel values in the feature vector
        }
    }

    return featureVector; // Return the feature vector
}

// Function to compute histogram of an image
cv::Mat computeHistogram(cv::Mat src){

    cv::Mat hist; // Initialize histogram matrix

    const int histsize = 16; // Histogram size

    // Building the histogram 
    hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1); // Initialize histogram matrix with zeros

    // Iterate through each pixel in the image
    for(int i=0;i<src.rows;i++){
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i); // Pointer to row i

        for(int j=0;j<src.cols;j++){

            float B = ptr[j][0]; // Blue channel value
            float G = ptr[j][1]; // Green channel value
            float R = ptr[j][2]; // Red channel value

            float divisor = R+G+B; // Total sum of RGB values

            divisor = divisor > 0.0 ? divisor : 1.0; // Avoid division by zero

            float r = R / divisor; // Normalized Red channel value
            float g = G / divisor; // Normalized Green channel value

            // Compute indexes r,g are in [0,1]
            int rindex = (int)( r * (histsize -1) + 0.5); // Index for Red channel
            int gindex = (int)( g* (histsize -1) + 0.5); // Index for Green channel

            // Increment the histogram
            hist.at<float>(rindex*histsize+gindex)++; // Update histogram bin
        }
    }

    // Normalize the histogram by the number of pixels
    hist /= (src.rows * src.cols); // Normalize histogram by the number of pixels
    

    return hist; // Return computed histogram
}

// Function to extract a single histogram feature vector from an image
vector<float> singleHM(cv::Mat &src){
    
    cv::Mat hist; // Initialize histogram

    hist  = computeHistogram(src); // Compute histogram of the image

    vector<float> featureVector; // Initialize feature vector
    for (int y = 0; y < hist.rows; ++y) {
        for (int x = 0; x < hist.cols; ++x) {
            featureVector.push_back(hist.at<float>(y, x)); // Store histogram values in the feature vector
        }
    }

    return featureVector; // Return the feature vector
}

// Function to extract a multi-region histogram feature vector from an image
vector<float> multiHM(cv::Mat &src){

    cv::Mat hist1, hist2; // Initialize histograms for whole image and center region

    // First histogram (whole image)
    hist1 = computeHistogram(src);

    // Second histogram (center 7x7 region)
    int x = (src.cols -7) / 2; // Calculate x-coordinate of the starting point for the center 7x7 region
    int y = (src.rows -7) / 2; // Calculate y-coordinate of the starting point for the center 7x7 region
    cv::Rect roi(x, y, 7,7); // Define a region of interest (ROI) for the center 7x7 region
    cv::Mat centerRegion = src(roi); // Extract the center 7x7 region from the image
    hist2 = computeHistogram(centerRegion); // Compute histogram of the center region

    vector<float> featureVector; // Initialize feature vector
    for (int y = 0; y < hist1.rows; ++y) {
        for (int x = 0; x < hist1.cols; ++x) {
            featureVector.push_back(hist1.at<float>(y, x)); // Store histogram values of whole image in the feature vector
        }
    }
    for (int y = 0; y < hist2.rows; ++y) {
        for (int x = 0; x < hist2.cols; ++x) {
            featureVector.push_back(hist2.at<float>(y, x)); // Store histogram values of center region in the feature vector
        }
    }

    return featureVector; // Return the feature vector
}

// Function that applies Sobel X filter to the image
int sobelX3x3(cv::Mat &src, cv::Mat &dst){
  
  // Ensure destination image has same size as source image
  dst.create(src.size(), CV_16SC3); // Create destination image with the same size as source image

  // Apply Sobel X filter to each pixel in the source image
  for(int i=1;i<src.rows-1;i++){
    for(int j=1;j<src.cols-1;j++){
      for(int k=0;k<src.channels();k++){ 

        // Apply Sobel X filter
        int sum = (-1)*src.at<cv::Vec3b>(i-1, j-1)[k] + \
          0*src.at<cv::Vec3b>(i-1, j)[k] \
          + src.at<cv::Vec3b>(i-1,j+1)[k] + (-2)*src.at<cv::Vec3b>(i, j-1)[k] + \
          0*src.at<cv::Vec3b>(i,j)[k] + 2*src.at<cv::Vec3b>(i, j+1)[k] + \
          (-1)*src.at<cv::Vec3b>(i+1, j-1)[k] + (0)* src.at<cv::Vec3b>(i+1, j)[k] + \
          src.at<cv::Vec3b>(i+1, j+1)[k];
        
        // Store result in destination image
        dst.at<cv::Vec3s>(i,j)[k] = sum;
        dst.convertTo(dst, CV_16SC3); // Convert destination image to 16-bit signed integer format

      }
    }
  }
   

  return 0; // Return success
}

// Function that applies Sobel Y filter to the image
int sobelY3x3(cv::Mat &src, cv::Mat &dst){
  
  // Ensure destination image has same size as source image
  dst.create(src.size(), CV_16SC3); // Create destination image with the same size as source image

  // Apply Sobel Y filter to each pixel in the source image
  for (int i = 1; i < src.rows - 1; i++) {
    for (int j = 1; j < src.cols - 1; j++) {
      for (int k = 0; k < src.channels(); k++) {

        // Apply Sobel Y filter
        int sum = (-1) * src.at<cv::Vec3b>(i - 1, j - 1)[k] + \
                  (-2) * src.at<cv::Vec3b>(i - 1, j)[k] + \
                  (-1) * src.at<cv::Vec3b>(i - 1, j + 1)[k] + (0) * src.at<cv::Vec3b>(i, j - 1)[k] + \
                  0 * src.at<cv::Vec3b>(i, j)[k] + 0 * src.at<cv::Vec3b>(i, j + 1)[k] + \
                  (1) * src.at<cv::Vec3b>(i + 1, j - 1)[k] + (2) * src.at<cv::Vec3b>(i + 1, j)[k] + \
                  src.at<cv::Vec3b>(i + 1, j + 1)[k];
        
        // Store result in destination image
        dst.at<cv::Vec3s>(i, j)[k] = sum;
        dst.convertTo(dst,CV_16SC3); // Convert destination image to 16-bit signed integer format

      }
    }
  }

  return 0; // Return success
}


// Function that generates a gradient magnitude image from Sobel X and Y filtered images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){
  
  // Ensure input images are of type CV_32SC3
  sx.convertTo(sx, CV_16SC3); // Convert Sobel X filtered image to 16-bit signed integer format
  sy.convertTo(sy, CV_16SC3); // Convert Sobel Y filtered image to 16-bit signed integer format

  // Ensure destination image has the same size as the input images
  dst.create(sx.size(), CV_8UC3); // Create destination image with the same size as the input images
  
  // Compute gradient magnitude image
  for(int i=0;i<sx.rows;i++){
    for(int j=0;j<sx.cols;j++){
      for(int k=0;k<sx.channels();k++){
      
        float I = sqrt(static_cast<float>(sx.at<cv::Vec3s>(i,j)[k] * sx.at<cv::Vec3s>(i,j)[k] + \
                                          sy.at<cv::Vec3s>(i,j)[k] * sy.at<cv::Vec3s>(i,j)[k])); // Compute gradient magnitude

        // To ensure magnitude is within the uchar range [0,255]
        I = std::min(255.0f, std::max(0.0f, I)); // Clip magnitude value within the range [0, 255]

        dst.at<cv::Vec3b>(i,j)[k] = static_cast<uchar>(I); // Store magnitude in destination image

      }
    }
  }

  return 0;  // Return success

}


// Function to extract histogram and texture features from an image
vector<float> featureHM(cv::Mat &src){

    cv::Mat cHist, tHist; // Initialize histograms for color and texture features

    // Color histogram of Whole image
    cHist = computeHistogram(src); // Compute color histogram of the image

    // Texture histogram of whole image
    cv::Mat sobelx, sobely, dst; // Initialize images for Sobel X, Sobel Y, and gradient magnitude
    sobelX3x3(src, sobelx); // Apply Sobel X filter to the image
    sobelY3x3(src, sobely); // Apply Sobel Y filter to the image
    magnitude(sobelx, sobely, dst); // Compute gradient magnitude image
    tHist = computeHistogram(dst); // Compute texture histogram of the gradient magnitude image
    

    vector<float> featureVector; // Initialize feature vector
    for (int y = 0; y < cHist.rows; ++y) {
        for (int x = 0; x < cHist.cols; ++x) {
            featureVector.push_back(cHist.at<float>(y, x)); // Store color histogram values in the feature vector
        }
    }
    for (int y = 0; y < tHist.rows; ++y) {
        for (int x = 0; x < tHist.cols; ++x) {
            featureVector.push_back(tHist.at<float>(y, x)); // Store texture histogram values in the feature vector
        }
    }

    return featureVector; // Return the feature vector
}

// Function to compute color histogram features of an image
vector<float> computeColorHistogram(cv::Mat& image) {
    // Initialize histogram bins
    const int histSize = 256; // Histogram size
    const int histDim = 3; // Histogram dimension (for RGB channels)
    float range[] = {0, 256}; // Range for histogram bins
    const float* histRange = {range}; // Range pointer

    // Split the image into channels
    std::vector<cv::Mat> bgr_planes; // Initialize vector to store channel images
    cv::split(image, bgr_planes); // Split the image into its RGB channels

    // Initialize histogram features vector
    std::vector<float> hist_features(histSize * histDim, 0.0f); // Initialize feature vector for histogram

    // Compute histogram for each channel
    for (int ch = 0; ch < histDim; ++ch) {
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                int bin = static_cast<int>(bgr_planes[ch].at<uchar>(i, j)); // Calculate histogram bin for pixel
                hist_features[ch * histSize + bin]++; // Increment bin count in the feature vector
            }
        }
    }

    // Normalize histogram
    int totalPixels = image.rows * image.cols; // Total number of pixels in the image
    for (int i = 0; i < histDim * histSize; ++i) {
        hist_features[i] /= totalPixels; // Normalize bin counts by the total number of pixels
    }

    return hist_features; // Return normalized histogram features
}
