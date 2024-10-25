/*
    Code to identify files in a directory
*/

#include<iostream>
#include<opencv2/opencv.hpp>
#include<dirent.h>

/*
    Given a directory on the command line, scans through the directory for image files.
    Prints out the full path name for each file. This can be used as an argument to
    fopen or to cv::imread.
*/

using namespace std;

int main(int argc, char *argv[]){

    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // Checking for sufficient arguments
    if(argc < 2){
        cout<<"Usage: <directory path>\n"<<argv[0];
        exit(-1);
    }

    // Get the directory path
    strcpy(dirname, argv[1]);
    cout<<"Processing directory: "<< dirname<<"\n";

    // Open the directory
    dirp = opendir(dirname);
    if(dirp==NULL){
        cout<<"Cannot open directory: "<< dirname<<"\n";
        exit(-1);
    }

    // Loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ){

        // Check if the file is an image
        if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ){

            cout<<"Processing image files: "<<dp->d_name<<"\n";

            // Build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            cout<<"Full path name: "<<buffer<<"\n";

        }

    }

    cout<<"Terminating\n";

    return 0;
}