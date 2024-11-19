# Use Ubuntu as the base image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Copy the application code into the container
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    ultralytics \
    opencv-python\
    opencv-python-headless \
    Flask \
    Pillow \
    matplotlib \
    numpy

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python3", "app.py"]



# # ubuntu version 
# FROM ubuntu:20.04

# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y \
#     python3.9 \
#     python3-pip \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python3.9 /usr/bin/python

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt


# COPY best.pt /app/best.pt
# COPY logo.png /app/logo.png
# COPY . /app

# EXPOSE 8080

# CMD ["python", "number_plate.py"]
