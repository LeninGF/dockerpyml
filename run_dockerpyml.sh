#!/bin/bash
docker run -it dockermlpy conda activate tfwin | python  mlapp.py --train --evaluate --save_model
