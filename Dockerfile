FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file (we'll create this)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY number_plate.py .
COPY best.pt .
COPY logo.png .

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["python", "number_plate.py"]