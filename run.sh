#!/bin/bash

# Ensure python environment is activated if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the single-file application
uvicorn app:app --reload --host 0.0.0.0 --port 7860