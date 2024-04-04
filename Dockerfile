# USING OFICIAL PYTHON IMAGE
# FROM python:3.10-slim
# USING OFFICIAL ANACONDA RUNTIME AS PARENT IMAGE
FROM continuumio/anaconda3
# SET A WORKING DIRECTORY
WORKDIR /app

COPY mlapp.py .
COPY requirements.txt .
COPY environment.yml .
COPY entrypoint.sh /app/entrypoint.sh

# USING PIP TO INSTALL PACKAGES
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# USING ANACONDA TO INSTALL PACKAGES
# RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN conda env create -f environment.yml
# SHELL [ "conda", "run", "-n", "tfwin", "/bin/bash", "-c" ]







