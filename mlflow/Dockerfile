# Сервер MLflow с драйвером PostgreSQL и клиентом S3.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.43.100
COPY create_bucket.py /create_bucket.py
