#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -l mem=1G
#$ -N test

#$ -V # Exporta las variables de entorno
#$ -t 1-801 # Setea el array de indices

#$ -pe neworte 1

module load miniconda
conda activate qutip


python argtest6PN.py $SGE_TASK_ID