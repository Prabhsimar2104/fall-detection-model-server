# Use Python 3.11 to ensure mediapipe compatibility
FROM python:3.11-slim

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system libs needed by opencv and mediapipe
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
