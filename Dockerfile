FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Define environment variables
# don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# Set ENTRYPOINT to always call the Python script
ENTRYPOINT ["python", "main.py"]

# CMD sets the default arguments to train the model
CMD ["train", "--data", "data-set.csv"]
