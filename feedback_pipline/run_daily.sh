#!/bin/bash

# Activate virtual environment
source /root/piper/src/python/.venv/bin/activate

# Define date and time
LOG_DATETIME=$(date +"%Y%m%d_%H%M%S")

# Run Python scripts with logs including date and time
python /root/piper/src/python/feedback_pipline/pipeline.py >> "/root/piper/src/python/feedback_pipline/logs/pipeline_${LOG_DATETIME}.log" 2>&1
python /root/piper/src/python/feedback_pipline/pipeline_female.py >> "/root/piper/src/python/feedback_pipline/logs/pipeline_female_${LOG_DATETIME}.log" 2>&1
