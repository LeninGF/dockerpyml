# README
## ENG
This repository is to study how to use docker with a python script that makes some machine learning
## SPA
Este repositorio tiene por objeto estudiar usar docker para un script de python que usa machine learning
## REFERENCES/REFERENCIAS

* https://www.youtube.com/watch?v=0UG2x2iWerk&t=1s
* https://github.com/patrickloeber/python-docker-tutorial

 docker run -it dockermlpy conda activate tfwin | python  mlapp.py --train --evaluate --save_model

 ## To Replicate conda environment

 1. split in two files one for conda in `yml` and other for pip in `txt`
 2. to create the anaconda environment use:
    conda list --explicit>ENV.txt
 3. to create the requirements.txt file use `conda env export >ENV.yml` from the activated environment. Then copy into a requirement.txt file the contents in pip
 4. Create the anaconda environment and activate the shell to the the target environment.
 5. Install conda packages
 6. Install pip packages
 7. hopefully enjoy

 ## To run
    
    docker run -it dockerpyml /bin/bash -c "source activate mlenv && python mlapp.py --train --evaluate --save_model"



    