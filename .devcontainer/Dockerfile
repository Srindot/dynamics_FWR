# Use an official NVIDIA CUDA 12 development image as the base.
# The 'devel' tag includes the full CUDA toolkit, compiler (nvcc), and debugging tools,
# which are essential for building Python packages that have C++/CUDA extensions.
# Ubuntu 22.04 is a stable, modern base OS.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies.
# - build-essential: Compilers (gcc, g++) and tools needed for building software.
# - python3.11, python3-pip, python3-venv: A modern version of Python and its package management tools.
# - git, wget: Standard tools for version control and downloading files.
# - Rendering libraries for MuJoCo's viewer: MuJoCo requires OpenGL libraries to render simulations.
#   libgl1-mesa-glx, libglew-dev, and libglfw3-dev are crucial for this.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for better security and to avoid permission issues.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser
WORKDIR /home/appuser
USER root
RUN apt-get update && apt-get install -y sudo \
    && echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER appuser

# Set up a virtual environment for Python packages.
RUN python3.11 -m venv /home/appuser/venv
ENV PATH="/home/appuser/venv/bin:$PATH"

# Upgrade pip and install Python packages from requirements.txt.
# Using a requirements file is a best practice for managing dependencies.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set the default command to an interactive bash shell.
CMD ["/bin/bash"]
