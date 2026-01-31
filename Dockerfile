# Dockerfile for Streamlit dashboard
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files
COPY . /app

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements-deploy.txt

# Expose Streamlit default port
EXPOSE 8501

# Environment
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Run Streamlit
CMD ["streamlit", "run", "dashboard.py", "--server.port", "${PORT}", "--server.address", "0.0.0.0"]
