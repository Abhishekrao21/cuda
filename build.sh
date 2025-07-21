#!/bin/bash
# Simple build script for CUDA Histogram Project

echo "CUDA Histogram Project Build Script"
echo "==================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found!"
    echo "Please install CUDA Toolkit first."
    exit 1
fi

# Check for OpenCV
if ! pkg-config --exists opencv4; then
    echo "Error: OpenCV not found!"
    echo "Please install OpenCV development libraries."
    exit 1
fi

echo "Building project..."
make all

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Next steps:"
    echo "1. Generate test data: make generate-data"
    echo "2. Run the program: make run"
    echo "3. View plots: make plot"
    echo ""
    echo "Or run the complete pipeline: make pipeline"
else
    echo "Build failed!"
    exit 1
fi
