# Set your desired base image
FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install libraries.
# Zip is needed to zip the output files.
RUN apt-get update --fix-missing && \
    apt-get install -y \
        software-properties-common git && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        libspatialindex-dev \
        python3.8 \
        python3.8-venv \
        xorg \
        zip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Update default python version.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# Setup virtual environment and install pip.
ENV VIRTUAL_ENV=/opt/.venv
RUN python3.8 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip wheel

# Install requirements.txt .
COPY ./submission/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"

# Copy only Track-2 files.
COPY . /SMARTS/competition/track2
#RUN pip install -e /SMARTS/competition/track2/train

# [Do Not Modify]
# Run entrypoint script.
ENTRYPOINT ["/bin/sh","/SMARTS/competition/track2/entrypoint.sh"]