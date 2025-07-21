# Parallel Histogram Calculation (CUDA)

## Overview

Compute grayscale pixel histograms for a batch of images in parallel on the GPU.

## Build & Run

```bash
cd src
make
./histogram_app ../data 512 512
cd ../output
python3 ../plot_histograms.py