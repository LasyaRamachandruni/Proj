FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/optical-projects-main

# System packages (build tools, git for optional pulls)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Base project requirements (Ray/RLlib, Gymnasium, dotenv, networkx)
COPY requirements-win-py312.txt ./
RUN pip install --no-cache-dir -r requirements-win-py312.txt

# Torch (CPU build)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

# Plotting + optional TF/Keras for the soft-failure detector
RUN pip install --no-cache-dir \
    matplotlib==3.10.7 \
    keras==3.4.1 \
    tensorflow==2.16.1

# Project sources
COPY . .

# Ensure the collaborative-rl package is importable
ENV PYTHONPATH=/opt/optical-projects-main/collaborative-rl:$PYTHONPATH

# Default entrypoint: run the DQN runner
CMD ["python", "collaborative-rl/DQN_runner.py"]
