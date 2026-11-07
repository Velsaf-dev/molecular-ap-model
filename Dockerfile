FROM python:3.11-slim

# System dependencies for RDKit, LightGBM, and OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and repository data
COPY . .

# Run inference and pipeline validation
CMD ["python", "run_solution.py"]
