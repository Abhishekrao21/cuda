#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include "histogram_kernel.h"

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./histogram_app <image_folder> <width> <height>" << endl;
        return -1;
    }
    string folder = argv[1];
    int width  = stoi(argv[2]);
    int height = stoi(argv[3]);

    vector<string> files;
    for (auto& p: fs::directory_iterator(folder)) {
        files.push_back(p.path().string());
    }
    int batch = files.size();

    // host buffers
    vector<unsigned char> h_images(batch * width * height);
    vector<unsigned int> h_hist(batch * 256);

    // load images to grayscale and resize
    for (int i = 0; i < batch; ++i) {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(width, height));
        memcpy(&h_images[i * width * height], img.data, width * height);
    }

    // allocate on device
    unsigned char* d_images;
    unsigned int*  d_hist;
    cudaMalloc(&d_images, batch * width * height);
    cudaMalloc(&d_hist,   batch * 256 * sizeof(unsigned int));

    cudaMemcpy(d_images, h_images.data(), batch * width * height, cudaMemcpyHostToDevice);

    // launch: one block per image
    int threads = 256;
    histogramKernel<<<batch, threads, 256 * sizeof(unsigned int)>>>(
        d_images, width, height, d_hist, batch
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_hist.data(), d_hist, batch * 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // write CSVs
    fs::create_directory("../output");
    for (int i = 0; i < batch; ++i) {
        string out = "../output/hist_" + to_string(i) + ".csv";
        ofstream ofs(out);
        for (int j = 0; j < 256; ++j)
            ofs << j << "," << h_hist[i * 256 + j] << "\n";
    }

    cudaFree(d_images);
    cudaFree(d_hist);
    return 0;
}