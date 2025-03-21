FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /service

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Copy only requirements for better caching
COPY requirements.txt /service/

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install -U flash-attn --no-build-isolation

# Copy the rest of the application code
COPY . /service/

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
