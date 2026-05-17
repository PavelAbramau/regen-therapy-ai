# 1. Use a clean Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first to leverage Docker caching
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the module's code
COPY . .

# 6. Ensure the expected data directories exist
RUN mkdir -p data/raw_test_images data/predictions weights

# 7. Set the default command to run the inference script
CMD ["python", "src/predict.py"]