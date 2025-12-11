#!/bin/bash
#SBATCH --job-name=bothello_cpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s336426@studenti.polito.it
#SBATCH --partition=cpu_sapphire      # Use 'cpu_sapphire' (Isola 2) or 'cpu_skylake'
#SBATCH --time=0-01:00:00             # Max time 
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (1 task = 1 process)
#SBATCH --cpus-per-task=1             # 1 CPU per task (Single thread)
#SBATCH --mem=32G                     # Memory required per node
#SBATCH --output=job_%j.out           # Standard output log

module purge
module load gcc/10.2.0

# Setup Workspace on BeeGFS (Scratch)
SOURCE_DIR=$SLURM_SUBMIT_DIR
WORK_DIR=$SCRATCH/job_${SLURM_JOB_ID}

mkdir -p $WORK_DIR

cp $SOURCE_DIR/bin $WORK_DIR/
cd $WORK_DIR

./cpu > sim_output.txt

# Move the output text file back to your original folder
cp sim_output.txt $SOURCE_DIR/

rm -rf $WORK_DIR