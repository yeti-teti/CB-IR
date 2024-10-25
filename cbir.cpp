/*

    File to run after getting the feature vectors of all images in a csv file

    Code to take the target image, feature type, feature vector files and the number of output images

    Note: Make sure to run the featureCompute.cpp first to store the feature vectors in the csv file

    To run:
        make cbir
        ./cbir <target_image> <feature_type> <feature_vector_file> <output_images>

        Feature Types:
            1. Baseline matching
            2. Single histogram
            3. Multi histogram
            4. Texture and Color
            5. Deep Network embeddings
            6. Compare DNN embeddings and Classic features
            7. Custom Design
        
        Extensions:
            1. DNN with color histogram
            2. DNN with magnitude

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

#include<iostream> // Input-output stream operations
#include<fstream> // File stream operations
#include<opencv2/opencv.hpp> // OpenCV library for computer vision tasks
#include<vector> // Standard template library for vectors
#include "features.hpp" // Custom header file for feature extraction
#include<dirent.h> // Directory handling
#include "csv_util.h" // Custom header file for CSV file handling
#include<cstring> // String manipulation operations



using namespace std;

int main(int argc, char *argv[]){
    char targetImage[256]; // Stores the path of the target image
    char featureVectorFile[256]; // Stores the path of the file containing feature vectors
    int featureType; // Stores the type of feature to be used for comparison
    int outputImages; // Number of output images to display
    
    // File pointer and directory pointer for file and directory handling
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // Checking for sufficient command line arguments
    if(argc < 5){
        cout << "Usage: " << argv[0] << " <target_image> <feature_type> <feature_vector_file> <output_images>" << endl;
        exit(-1);
    }

    // Retrieving command line arguments
    strcpy(targetImage, argv[1]); // Target image path
    featureType = atoi(argv[2]); // Feature type for comparison
    strcpy(featureVectorFile, argv[3]); // Feature vector file path
    outputImages = atoi(argv[4]); // Number of output images to display
    
    // Type alias for the function pointer type for calculating distance between feature vectors
    using DistanceFunction = float (*)(const vector<float>&, const vector<float>&);
    DistanceFunction distanceM; // Variable of type DistanceFunction for holding different distance calculation functions
    
    // Reading feature vectors from file
    vector<char *> filenames; // Vector to store filenames
    vector<vector<float>> data; // Vector to store feature vectors
    if(read_image_data_csv(featureVectorFile, filenames, data)!=0){
        cout<<"Error: Unable to read feature vector file"<<endl;
        exit(-1);
    }

    // Reading the target image
    cv::Mat tImage = cv::imread(targetImage); // Reading target image
    if(tImage.empty()){
        cout<<"Error: Unable to read image:"<<targetImage<<endl;
    }

    // Computing the features for the target image
    char completeFN[256]; // Stores complete file name
    vector<float> targetFeatures; // Vector to store feature vector of target image
    switch(featureType){
        case 1: // Baseline feature extraction
            targetFeatures = baseline(tImage);
            distanceM = &SSD; // Sum of Squared Differences
            break;
        case 2: // Single Histogram feature extraction
            targetFeatures = singleHM(tImage);
            distanceM = &histIntersection; // Histogram Intersection
            break;
        case 3: // Multi Histogram feature extraction
            targetFeatures = multiHM(tImage);
            distanceM = &histIntersection; // Histogram Intersection
            break;
        case 4: // Feature Histogram feature extraction
            targetFeatures = featureHM(tImage);
            distanceM = &histIntersection; // Histogram Intersection
            break;
        case 5: // Custom feature extraction
            for(size_t i=0; i<filenames.size(); ++i){
                strcpy(completeFN, "olympus");
                strcat(completeFN, "/");
                strcat(completeFN, filenames[i]);

                if(strlen(filenames[i])==0){
                    cout<<"Error: Unable to read image:"<<completeFN<<endl;
                    continue;
                }

                if(strcmp(completeFN,targetImage) == 0){
                    targetFeatures = data[i];
                }
            }
            distanceM = &cosineDistance; // Cosine Distance
            break;
        case 7: // Color Histogram feature extraction
            targetFeatures = computeColorHistogram(tImage);
            distanceM = &histIntersection; // Histogram Intersection
            break;
        case 8: // Custom feature extraction
            for(size_t i=0; i<filenames.size(); ++i){
                strcpy(completeFN, "olympus");
                strcat(completeFN, "/");
                strcat(completeFN, filenames[i]);

                if(strlen(filenames[i])==0){
                    cout<<"Error: Unable to read image:"<<completeFN<<endl;
                    continue;
                }

                if(strcmp(completeFN,targetImage) == 0){
                    targetFeatures = data[i];
                }
            }
            distanceM = &histIntersection; // Histogram Intersection
            break;
        case 9: // Custom feature extraction
            for(size_t i=0; i<filenames.size(); ++i){
                strcpy(completeFN, "olympus");
                strcat(completeFN, "/");
                strcat(completeFN, filenames[i]);

                if(strlen(filenames[i])==0){
                    cout<<"Error: Unable to read image:"<<completeFN<<endl;
                    continue;
                }

                if(strcmp(completeFN,targetImage) == 0){
                    targetFeatures = data[i];
                }
            }
            distanceM = &SSD; // Sum of Squared Differences
            break;
        default:
            cout<<"Invalid feature type"<<endl;
            exit(-1);
    }
    
    // Calculating distance and sorting
    vector<pair<string, float>> distances; // Vector to store filename-distance pairs
    for(size_t i=0; i<filenames.size(); ++i){
        float distance = distanceM(targetFeatures, data[i]); // Computing distance
        distances.emplace_back(filenames[i], distance); // Storing filename-distance pair
    }

    // Sorting filenames based on distance
    sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    // Displaying top N matching images
    char buffer[256]; // Buffer for storing filename
    cv::Mat src; // Mat object for image storage
    
    cout<<"Top "<<outputImages<<" matches: "<<endl;
    for(int i=0;i<outputImages && i<distances.size(); i++){
        cout<<distances[i].first<<" (Distance: "<< distances[i].second << ")"<<endl;

        strcpy(buffer, "olympus");
        strcat(buffer, "/");
        const char *str = distances[i].first.c_str();
        strcat(buffer, str);

        src = cv::imread(buffer); // Reading image
        cv::imshow(buffer,src); // Displaying image
        cv::waitKey(0); // Waiting for key press
        cv::destroyWindow(buffer); // Destroying window
    }

    return 0;
}
