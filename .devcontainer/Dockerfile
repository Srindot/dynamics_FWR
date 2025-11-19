# Use an official NVIDIA CUDA 12 development image as the base.
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- UTF-8 CONFIGURATION START ---
# Set the locale environment variables so Python and terminal tools don't fail on Unicode.
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
# --- UTF-8 CONFIGURATION END ---

# Install system dependencies.
# Added 'locales' to the list and added a step to generate the UTF-8 locale.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    locales \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    patchelf \
    mesa-utils \
    x11-apps \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for better security and to avoid permission issues.
RUN useradd --create-home --shell /bin/bash container_usr
USER container_usr
WORKDIR /home/container_usr

# Switch to root temporarily to give the user sudo access (optional but often useful in devcontainers)
USER root
RUN apt-get update && apt-get install -y sudo \
    && echo "container_usr ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER container_usr

# Set up a virtual environment for Python packages.
RUN python3.11 -m venv /home/container_usr/venv
ENV PATH="/home/container_usr/venv/bin:$PATH"

# Upgrade pip and install Python packages from requirements.txt.
COPY --chown=container_usr:container_usr requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set the default command to an interactive bash shell.
CMD ["/bin/bash"]