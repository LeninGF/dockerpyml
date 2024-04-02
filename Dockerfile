FROM python:3.10-slim

WORKDIR /usr/ml

COPY mlapp.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt




