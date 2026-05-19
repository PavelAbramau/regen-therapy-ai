# 1. Use a clean Python image
FROM python:3.10-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy only the requirements first
COPY requirements.txt .

# 4. Install standard libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Clean install OpenCV Headless
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless

# 6. Copy all your code and models into the container
COPY . .

# 7. Set the default command
CMD ["python", "analyze_wound.py", "--help"]
