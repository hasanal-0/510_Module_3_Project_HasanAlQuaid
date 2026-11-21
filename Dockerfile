# AIPI 510: Module 3 Project - Dockerfile
# Hasan Al-Quaid
# Packages the API so it runs the same on any computer or on Cloud Run.

# same Python version the model was trained with
FROM python:3.11-slim

# folder inside the container where the project lives
WORKDIR /app

# install the packages first, so Docker can reuse this step when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the project code (.dockerignore keeps out data, models, MLflow files, etc.)
COPY . .

# Cloud Run sends requests to port 8080
EXPOSE 8080

# serve entry point (default): start the API
# train entry point: docker run IMAGE python src/train.py
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}
