
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model at build time
RUN python -m src.train

# Default command: run prediction
ENTRYPOINT ["python", "main.py"]
