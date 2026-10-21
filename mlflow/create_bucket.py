"""Создать bucket для артефактов MLflow, если его ещё нет."""
import os

import boto3

bucket = os.environ["S3_BUCKET"]
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
if bucket not in [b["Name"] for b in s3.list_buckets()["Buckets"]]:
    s3.create_bucket(Bucket=bucket)
print(f"bucket {bucket} ready")
