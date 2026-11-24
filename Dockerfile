# Start with Python in a small Linux environment.
FROM python:3.13-slim

# Show Python output immediately and avoid creating bytecode files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working folder inside the image.
WORKDIR /app

# Install the project's dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application package.
COPY abstract_classifier/ ./abstract_classifier/

# Document the port used by the API.
EXPOSE 8000

# Start the API when the container runs.
CMD ["python", "-m", "uvicorn", "abstract_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]