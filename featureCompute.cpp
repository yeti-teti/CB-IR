/*
   Compute the features of each image in the specified directory and store them in a CSV file.

   To run:
        make featureCompute
        ./featureCompute <directory path> <feature type> <csv file>

        Feature Types:
            1. Baseline matching
            2. Single histogram
            3. Multi histogram
            4. Texture and Color
            5. Deep Network embeddings
            6. Compare DNN embeddings and Classic features
            7. Custom Design
                a. Color histogram to get combined color histogram from all channels instead of single channels like Task 2 and 3
        
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

using namespace std;

int main(int argc, char *argv[]){
    char dirname[256]; // Stores the directory path
    char buffer[256]; // Buffer for constructing file paths
    char csvfileName[256]; // Stores the name of the CSV file to write feature vectors
    int featureType; // Type of feature to be computed
    
    // File and directory handling variables
    fstream csvFile;
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // Checking for sufficient command line arguments
    if(argc < 4){
        cout<<"Usage: "<<argv[1]<<"<directory path> <feature type> <csv file>\n";
        exit(-1);
    }

    // Retrieving command line arguments
    strcpy(dirname, argv[1]); // Directory path
    cout<<"Processing directory: "<< dirname<<"\n";

    // Open the directory
    dirp = opendir(dirname);
    if(dirp==NULL){
        cout<<"Cannot open directory: "<< dirname<<"\n";
        exit(-1);
    }

    // Get feature type
    featureType = atoi(argv[2]); // Feature type for computation

    // Get the csv file name
    strcpy(csvfileName, argv[3]); // CSV file name for storing feature vectors

    // Read feature vectors from file
    vector<char *> filenames; // Vector to store filenames
    vector<vector<float>> data; // Vector to store feature vectors
    char resNet[256]; // File name for feature vectors
    strcpy(resNet, "ResNet18_olym.csv"); // Assuming default name for feature vector file
    if(read_image_data_csv(resNet, filenames, data)!=0){ // Reading feature vectors
        cout<<"Error: Unable to read feature vector file"<<endl;
        exit(-1);
    }

    char completeFN[256]; // Stores complete file name
    vector<float> featureVector; // Vector to store feature vector
    vector<float> featureVector_2; // Additional vector for combining features
    vector<float> featureVector_3;
    vector<float> featureVector_4;

    // Loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ){

        // Check if the file is an image
        if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") || strstr(dp->d_name, ".jpeg")){

            // Build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // Read the image
            cv::Mat image = cv::imread(buffer);
            if(image.empty()){
                cout<<"Error: Unable to read image:"<<buffer<<endl;
                continue;
            }

            // Compute the feature for each image based on the specified feature type
            switch(featureType){
                case 1: // Baseline feature extraction
                    featureVector = baseline(image);
                    break;
                case 2: // Single Histogram feature extraction
                    featureVector = singleHM(image);
                    break;
                case 3: // Multi Histogram feature extraction
                    featureVector = multiHM(image);
                    break;
                case 4: // Feature Histogram feature extraction
                    featureVector = featureHM(image);
                    break;
                case 7: // Color Histogram feature extraction
                    featureVector = computeColorHistogram(image);
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

                        if(strcmp(completeFN,buffer) == 0){
                            featureVector_2 = data[i];
                        }
                    }
                    featureVector = computeColorHistogram(image);
                    featureVector.insert(featureVector.end(), featureVector_2.begin(), featureVector_2.end());
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

                        if(strcmp(completeFN,buffer) == 0){
                            featureVector_2 = data[i];
                        }
                    }

                    featureVector = featureHM(image);
                    featureVector.insert(featureVector.end(), featureVector_2.begin(), featureVector_2.end());
                    break;
                default:
                    cout<<"Invalid feature type"<<endl;
                    break;
            }

            // Append the feature vector to the CSV file
            append_image_data_csv(csvfileName, dp->d_name, featureVector, 0);

        }

    }

    cout<<"Terminating\n";

    return 0;
}
