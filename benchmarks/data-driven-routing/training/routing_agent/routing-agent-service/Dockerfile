FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime


WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
COPY *.py .

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

ENV PORT=8080
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LOG_LEVEL=INFO
# ENV MODEL_PATH=/app/latency_predictor.joblib

EXPOSE 8080

CMD ["python", "routing_agent_service.py"]