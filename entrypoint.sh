#!/bin/bash
conda run --no-capture-output -n tfwin python mlapp.py --train --evaluate --save_model
