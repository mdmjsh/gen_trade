#!/bin/sh
# poetry run python test_s3.py
PARALLEL=1 poetry run python genetic.py --write_s3=True --write_local=True --fitness_function=p