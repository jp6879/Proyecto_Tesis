#!/bin/bash

# Script para correr trabajo serial

# Directorio actual es el raiz
#$ -cwd

# Nombre del proceso
#$ -N m2TrainNODE

# stdout y stderr al mismo archivo de salida
#$ -j y

# Usar bash como shell para los comandos que se ejecutaran
#$ -S /bin/bash

# Pido la cola a usar
#$ -q copahue

# Pido 64GB RAM para el proceso (obligatorio)
#$ -l mem=64G

# Las variables de entorno actual son incluidas en el trabajo
#$ -V

#$ -t 1-18 # Setea el array de indices

# Llamamos al modulo que tiene julia, y ejecutamos el programa tal cual lo llamarķamos desde la linea
# de comandos con los argumentos correspondientes al array de indices

module load julia-1.9.0

julia mini_Exploracion_Paralelizada_NODE_Minibatch.jl $SGE_TASK_ID

