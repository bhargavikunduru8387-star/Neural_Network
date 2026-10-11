
FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir -r requirements-docker.txt
COPY app.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
