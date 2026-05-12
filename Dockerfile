FROM python:3.11-slim

# Working directory
WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++

# Install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt


# Copy application files
COPY app.py .
COPY configuration ./configuration
COPY models ./models
COPY outputs ./outputs

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]