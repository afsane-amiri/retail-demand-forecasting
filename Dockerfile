# Use a lightweight Python 3.10 image.
FROM python:3.10-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy dependency definitions first.
# Docker can cache this layer when application code changes.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code.
COPY app/ ./app/

# Copy the exported production model.
COPY models/production_model/ ./models/production_model/

# Expose the port used by FastAPI/Uvicorn.
EXPOSE 8000

# Start the API when the container starts.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]