FROM python:3.10-slim

# System deps for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/ models/

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
