FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r /workspace/requirements.txt

RUN mkdir -p /workspace/notebooks \
    /workspace/ksl_project_data/logs \
    /workspace/ksl_project_data/features \
    /workspace/ksl_project_data/models \
    /workspace/.fiftyone

EXPOSE 8888
EXPOSE 5151

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=", \
     "--NotebookApp.password=", \
     "--notebook-dir=/workspace"]