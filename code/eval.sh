#!/bin/bash -l

#$ -P seizuredet

#$ -l h_rt=0:40:00

#$ -pe omp 2

#$ -l gpus=2

#$ -l gpu_c=6.0

#$ -j y

#$ -m ea

#$ -N Sz-challenge-pipeline


conda activate szcore


python code/pipeline.py #$SGE_TASK_ID


conda deactivate