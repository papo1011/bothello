#!/bin/bash
#SBATCH --job-name=bothello_cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s336426@studenti.polito.it
#SBATCH --partition=cpu_sapphire      # Use 'cpu_sapphire' or 'cpu_skylake'
#SBATCH --time=0-01:00:00             # Max time 
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (1 task = 1 process)
#SBATCH --cpus-per-task=1             # 1 CPU per task (Single thread)
#SBATCH --mem=32G                     # Memory required per node
#SBATCH --output=job_%j.out           # Standard output log

module purge
module load gcc/12.4.0

g++ -O3 -Wall -o cpu main.cpp src/*.cpp

./cpu > $SLURM_SUBMIT_DIR/sim_output.txt
