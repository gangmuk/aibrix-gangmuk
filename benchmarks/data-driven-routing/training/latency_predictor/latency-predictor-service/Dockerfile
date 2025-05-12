FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY latency_predictor_service.py .
COPY latency_predictor.joblib .

ENV PORT=8080

EXPOSE 8080

CMD ["python", "latency_predictor_service.py"]