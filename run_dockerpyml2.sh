#!/bin/bash
docker run -it dockermlpy /bin/bash -c "source activate mlenv && python mlapp.py --train --evaluate --save_model" >> report2.log 2>&1
