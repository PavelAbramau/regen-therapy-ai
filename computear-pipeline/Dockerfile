# syntax=docker/dockerfile:1
# 1. Use a clean Python image
FROM python:3.10-slim

# 2. Install git (Required for pip to install from github repos)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy only the requirements first
COPY requirements.txt .


RUN --mount=type=secret,id=github_token \
    git config --global url."https://$(cat /run/secrets/github_token)@github.com/".insteadOf "https://github.com/" && \
    pip install --no-cache-dir -r requirements.txt

# 6. Rip out everything OpenCV-related, then cleanly install ONLY the headless version
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# 7. Copy all your code and models into the container
COPY . .

# 8. Set the default command
CMD ["python", "analyze_wound.py", "--help"]
