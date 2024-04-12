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
COPY tfmlenv.txt .
COPY requirements.txt .

# USING ANACONDA TO INSTALL PACKAGES

RUN /opt/conda/bin/conda update -n base -c defaults conda
# RUN conda create --name mlenv --file mlenv.txt
RUN  conda create --name tfmlenv --file tfmlenv.txt

# Activate the conda environment and install pip packages from the requirements.txt file
# SHELL ["conda", "run", "-n", "mlenv", "/bin/bash", "-c"]
SHELL ["conda", "run", "-n", "tfmlenv", "/bin/bash", "-c"]
# RUN pip install -r requirements.txt

# Make sure the environment is activated:
# RUN echo "source activate mlenv" >> ~/.bashrc
RUN echo "source activate tfmlenv" >> ~/.bashrc

# Assuming your environment activation and Python script are part of the Docker image,
# you can specify them in the CMD instruction.
# CMD ["bash", "-c", "source activate tfmlenv && python mlapp.py --train --evaluate --save_model --read_sql"]





