# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Expose port for Flask
EXPOSE 5000

# Run the app
CMD ["python", "image_search_engine.py"]
