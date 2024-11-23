# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Builds ultralytics/yolov5:latest-cpu image on DockerHub https://hub.docker.com/r/ultralytics/yolov5
# Image is CPU-optimized for ONNX, OpenVINO and PyTorch YOLOv5 deployments

# Start FROM Ubuntu image https://hub.docker.com/_/ubuntu
FROM ubuntu:20.04

# Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
# g++ required to build 'tflite_support' and 'lap' packages, libusb-1.0-0 required for 'tflite_support' package
#RUN apt-get update \
#    && apt-get install --no-install-recommends -y python3-pip git zip curl htop libgl1-mesa-dev libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0
# RUN alias python=python3

RUN apt-get update && apt-get upgrade -y
RUN apt-get install --no-install-recommends -y python3-pip git zip curl
    # libgl1-mesa-dev libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0
RUN apt-get install -y libgl1-mesa-dev libxrandr2 libxcomposite1 libxtst6
RUN apt-get install -y libxkbcommon-dev libdbus-1-dev libasound2-dev
RUN apt-get install -y x11-apps python3-tk

# Remove python3.11/EXTERNALLY-MANAGED or use 'pip install --break-system-packages' avoid 'externally-managed-environment' Ubuntu nightly error
RUN rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED

# Install pip packages
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt albumentations gsutil notebook \
    coremltools onnx onnx-simplifier onnxruntime 'openvino-dev>=2023.0' \
    # tensorflow tensorflowjs \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

CMD ["python", "main.py"]
