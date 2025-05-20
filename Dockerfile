# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgtk2.0-dev \
        wget \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirement.txt /app/requirement.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirement.txt

# Copy project files
COPY . /app

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]