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
# RUN conda install --yes --file requirements.txt
RUN conda env create -f environment.yml
# RUN chmod +x /app/entrypoint.sh
SHELL [ "conda", "run", "-n", "tfwin", "/bin/bash", "-c" ]
# RUN python -c "import tensorflow as tf"

ENTRYPOINT ["conda", "run", "-n", "tfwin", "python", "mlapp.py", "--train", "--evaluate", "--save_model"]
# ENTRYPOINT [ "/app/entrypoint.sh" ]





