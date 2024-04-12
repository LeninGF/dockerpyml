#!/bin/bash
docker run -it --rm --env-file /path/to/your/.env -v /path/to/your/outputs:/falconiel/app/outputs dockerpyml /bin/bash -c "source activate tfmlenv && python mlapp.py --train --evaluate --save_model --read_sql"
