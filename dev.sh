#!/bin/bash

# Activate the virtual environment
source ./dev-venv/bin/activate

# Run the video inference script from src directory
# Pass all arguments through, with --headless as default for server environments
cd src && python video_inference_ncnn.py