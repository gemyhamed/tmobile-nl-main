FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Default command (train as the default mode)
CMD ["python", "main.py", "train", "--data", "./data/training_data.csv"]
