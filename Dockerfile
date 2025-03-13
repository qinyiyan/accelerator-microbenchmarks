# Use a base image with Python and Git
FROM python:3.10-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/qinyiyan/accelerator-microbenchmarks.git

# Navigate to the repository directory
WORKDIR /app/accelerator-microbenchmarks

# Install dependencies
RUN pip install --upgrade pip && \
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
    pip install --upgrade clu tensorflow tensorflow-datasets && \
    pip install jsonlines && \
    pip install ray[default]

# Set environment variables
ENV JAX_PLATFORMS=tpu,cpu \
    ENABLE_PJRT_COMPATIBILITY=true

# Optional: Expose a port if your application uses one
# EXPOSE 8080

# Optional: Define the command to run your application
# CMD ["python", "your_script.py"]
