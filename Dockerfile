# USING OFICIAL PYTHON IMAGE
# FROM python:3.10-slim
# USING OFFICIAL ANACONDA RUNTIME AS PARENT IMAGE
FROM continuumio/anaconda3
# FROM mambaorg/micromamba
# SET A WORKING DIRECTORY
WORKDIR /falconiel/app

COPY mlapp.py .
COPY src /falconiel/app/src
# COPY mlenv.yml .
COPY mlenv.txt  .
COPY requirements.txt .
# COPY environment3.yml .
# COPY entrypoint.sh /app/entrypoint.sh

# USING PIP TO INSTALL PACKAGES
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# USING ANACONDA TO INSTALL PACKAGES
# RUN /opt/conda/bin/conda activate
RUN /opt/conda/bin/conda update -n base -c defaults conda
# RUN conda env create -f mlenv.yml 
RUN conda create --name mlenv --file mlenv.txt

# Activate the conda environment and install pip packages from the requirements.txt file
SHELL ["conda", "run", "-n", "mlenv", "/bin/bash", "-c"]
RUN pip install -r requirements.txt

# Make sure the environment is activated:
RUN echo "source activate mlenv" >> ~/.bashrc

# RUN mamba env create -f environment3.yml
# RUN mamba install --yes --file environment3.yml 
#&& micromamba clean --all --yes
# SHELL [ "conda", "run", "-n", "tfwin", "/bin/bash", "-c" ]
# Add the 'conda activate' command to .bashrc
# RUN echo "conda activate mlenv" >> ~/.bashrc






