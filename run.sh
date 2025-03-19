#!/bin/bash

# Ensure python environment is activated if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the single-file application
uvicorn app.main:app --reload --host 0.0.0.0 --port 7860