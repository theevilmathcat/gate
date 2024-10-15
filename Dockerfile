# Use the official Python image from the Docker Hub
FROM python:3.10

# Install OpenGL dependencies for OpenCV and PostgreSQL client
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 postgresql-client

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

ENV PIP_DEFAULT_TIMEOUT=100

# Install the required Python packages
RUN pip install --no-cache-dir --timeout 100 -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Create necessary directories
RUN mkdir -p /app/model_inputs/employee_photos /app/model_outputs /app/training_classes /app/static/test_images

# Expose the port your app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]