#!/bin/bash
/usr/bin/docker run -i dockermlpy /bin/bash -c "source activate mlenv && python mlapp.py --train --evaluate --save_model"
