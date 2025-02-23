#!/bin/bash -l

#$ -P seizuredet

#$ -l h_rt=16:00:00

#$ -t 1-10

#$ -pe omp 4

#$ -l gpus=2

#$ -l gpu_c=6.0

#$ -j y

#$ -m ea

#$ -N Sz-challenge-training


module load python3/3.10.12


module load pytorch/1.13.1


python code/run_train.py $SGE_TASK_ID
