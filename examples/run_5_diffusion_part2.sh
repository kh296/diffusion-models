#!/bin/bash -l
#SBATCH --job-name=diffusion400 # create a short name for the job
#SBATCH --output=%x_%j.log      # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=2               # number of nodes
#SBATCH --gres=gpu:4            # number of allocated gpus per node
#SBATCH --time=02:00:00         # total run time limit (HH:MM:SS)
#
# Script for running pytorch example,
# based around diffusion models and MNIST dataset,
# with use of distributed data parallel.
#
# It's assumed that the environment for running pytorch applications
# can be set up with:
# source ../envs/diffusion-models-setup.sh
#
# This script can be run interactively:
#     ./run_5_diffusion_part2.sh
# or can be submitted to a Slurm batch system, substituting
# valid project account for <project_account>:.
#     sbatch --acount=<project_account> run_5_diffusion_part2.sh
#
# Before running, or inside this script, the command used to perform
# distributed processing may be chosen by setting the environment variable
# LAUNCH_MODE to one of "mpiexec" (default), "srun", "torchrun".

T1=${SECONDS}
echo "Job start on $(hostname): $(date)"

# Exit at first failure.
set -e

# Ensure that Slurm environment variables are set,
# also if outside of a Slurm environment.
#
# The variables specifying number of tasks per node (SLURM_NTASKS_PER_NODE),
# number of tasks (SLURM_NTASKS), and number of CPU cores per task
# (SLURM_CPUS_PER_TASK) are used in defining task parallelisation.
# These numbers depend on the number of nodes allocated, on the
# number of GPUs per node, and on how GPUs are configured.
#
# On Dawn, if a single node is allocated, then it may be allocated with
# 1, 2, 3, or 4 GPUs (but not with 0 GPUs).  If more than one node is
# allocated, all must be allocated with all (4) GPUs.  If GPUs are
# used in "FLAT" mode, the two stacks of each GPU are treated as two
# root devices.  If GPUs are used in "COMPOSITE" mode, the two stacks
# of each GPU are treated as a single root device.  For more information
# about modes for Intel GPUs, see:
# https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/exposing-device-hierarchy.html
# https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
#
# The variables specifying node(s) allocated (SLURM_JOB_NODELIST)
# and job id (SLURM_JOB_ID) are used to define the master address and
# port for task parallelisation.

# Perform environment setup.
WORKSHOP_HOME=$(cd $(dirname "$0")/..; pwd)
if [[ ${WORKSHOP_HOME} == /var/spool/* ]]; then
    WORKSHOP_HOME=$(dirname $(pwd))
fi

if [ -z "${ENV_NAME}" ]; then
    export ENV_NAME="diffusion-models"
fi
SETUP_SCRIPT="${WORKSHOP_HOME}/envs/${ENV_NAME}-setup.sh"
SETUP="source ${SETUP_SCRIPT}"
echo ""
echo ${SETUP}
${SETUP}

# Default to 1 node allocated if running outside of Slurm.
if [[ -z "${SLURM_NNODES}" ]]; then
    SLURM_NNODES=1
fi

# Determine number of root devices per GPU on Dawn.
if [[ "FLAT" == ${ZE_FLAT_DEVICE_HIERARCHY} ]]; then
    DEVICES_PER_GPU=2
else
    DEVICES_PER_GPU=1
fi

# Determine number of tasks per node, with one task per GPU root device,
# or defaulting to 1 if there are no GPUs.
# Where GPUs are present, set affinity mask to match number of root devices.
if [[ -z "${SLURM_GPUS_ON_NODE}" ]]; then
    SLURM_NTASKS_PER_NODE=1
else
    SLURM_NTASKS_PER_NODE=$((${SLURM_GPUS_ON_NODE}*${DEVICES_PER_GPU}))
    if [[ ${SLURM_NTASKS_PER_NODE} -gt 1 ]]; then
        export ZE_AFFINITY_MASK=$(seq -s, 0 $((${SLURM_NTASKS_PER_NODE}-1)))
    else
        export ZE_AFFINITY_MASK=0
    fi
fi

# Determine total number of tasks.
SLURM_NTASKS=$((${SLURM_NNODES}*${SLURM_NTASKS_PER_NODE}))

# Determine number of CPU cores per task.
if [[ -z "${SLURM_CPUS_ON_NODE}" ]]; then
    SLURM_CPUS_ON_NODE=1
fi
export SLURM_CPUS_PER_TASK=$((${SLURM_CPUS_ON_NODE}/${SLURM_NTASKS_PER_NODE}))
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}\

# Ensure that value assigned to SLURM_JOB_NODELIST.
if [[ -z "${SLURM_JOB_NODELIST}" ]]; then
    SLURM_JOB_NODELIST=$(hostname)
fi

# Ensure that value assigned to SLURM_JOB_ID.
if [[ -z "${SLURM_JOB_ID}" ]]; then
    SLURM_JOB_ID=5100
fi

# Unset and set Slurm variables for compatibility with srun.
unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_NODE
SLURM_EXPORT_ENV=ALL

# Create list of host names.
if command -v scontrol 1>/dev/null 2>&1; then
    HOSTS="$(echo $(scontrol show hostnames ${SLURM_JOB_NODELIST})\
        | sed 's/ /,/g')"
    DIST_URL="${HOSTS%%,*}"
else
    HOSTS="${SLURM_JOB_NODELIST}"
    DIST_URL="127.0.0.1"
    DIST_URL="localhost"
fi
HOSTS=$(echo "${HOSTS}" | cut -d',' -f1-"${SLURM_NNODES}")
DIST_PORT=$(( (SLURM_JOB_ID % 10000) + 50000 ))
echo ""
echo "Node(s) used:"
echo "${HOSTS}"

# Define launch commands, and perform initial module imports on each node.
# (Initial imports can be slow.  They aren't essential here, and are performed
# before running the application only so that so that the time for the initial
# imports isn't included in the application timing.)

# Allow for torch application to be launched using any of:
# mpiexec, srun, torchrun.
if [[ -z "${LAUNCH_MODE}" ]]; then
    LAUNCH_MODE="mpiexec"
fi

PYTHON_LAUNCH="python 5_diffusion_part2.py"
PYTHON_IMPORT_LAUNCH="python -c 'import torch; import torchvision;'"
TORCHRUN_LAUNCH="torchrun --nnodes ${SLURM_NNODES} --nproc_per_node ${SLURM_NTASKS_PER_NODE} --rdzv-backend=c10d --rdzv-endpoint=${DIST_URL}:${DIST_PORT} ./5_diffusion_part2.py"
SRUN_LAUNCH="srun --nodes=${SLURM_NNODES} --ntasks-per-node=${SLURM_NTASKS_PER_NODE} --gres=gpu:${SLURM_GPUS_ON_NODE}"
SRUN1_LAUNCH="srun --nodes=${SLURM_NNODES} --ntasks-per-node=1 --gres=gpu:${SLURM_GPUS_ON_NODE}"
MPI_LAUNCH="mpiexec -n ${SLURM_NTASKS}"
MPI1_LAUNCH="mpiexec -n ${SLURM_NNODES}"

if [[ "torchrun" == "${LAUNCH_MODE}" ]]; then
    if [[ "${SLURM_NNODES}" -gt 1 ]]; then
        LAUNCH="${SRUN1_LAUNCH} ${TORCHRUN_LAUNCH}"
	IMPORT_LAUNCH="${SRUN1_LAUNCH} ${PYTHON_IMPORT_LAUNCH}"
    else
        LAUNCH="${TORCHRUN_LAUNCH}"
        IMPORT_LAUNCH="${PYTHON_IMPORT_LAUNCH}"
    fi
elif [[ "srun" == "${LAUNCH_MODE}" ]]; then
    LAUNCH="${SRUN_LAUNCH} ${PYTHON_LAUNCH}"
    IMPORT_LAUNCH="${SRUN1_LAUNCH} ${PYTHON_IMPORT_LAUNCH}"
elif [[ "mpiexec" == "${LAUNCH_MODE}" ]]; then
    if [[ $(mpiexec --version) == *"Open MPI"* ]]; then
        HOSTS=$(echo "${HOSTS}" | tr "," "\n" | sed "s/$/:${SLURM_NTASKS_PER_NODE}/" | paste -sd,)
        MPI_LAUNCH+=" -N ${SLURM_NTASKS_PER_NODE} --host ${HOSTS}"
        MPI1_LAUNCH+=" -N 1 --host ${HOSTS}"
    else
        MPI_LAUNCH+=" -ppn ${SLURM_NTASKS_PER_NODE} --hosts ${HOSTS}"
        MPI1_LAUNCH+=" -ppn 1 --hosts ${HOSTS}"
    fi
    LAUNCH="${MPI_LAUNCH} ${PYTHON_LAUNCH}"
    IMPORT_LAUNCH="${MPI1_LAUNCH} ${PYTHON_IMPORT_LAUNCH}"
fi

# Initial package import can be slow.  Perform before running
# application, so that the initial time isn't included in
# the application timing.
echo ""
if [[ "${SLURM_NNODES}" -gt 1 ]]; then
    IMPORT_INFO=" on each node"
else
    IMPORT_INFO=""
fi
echo "Performing initial import of torch${IMPORT_INFO}"
T2=${SECONDS}
CMD="${IMPORT_LAUNCH}"
echo "${CMD}"
eval "${CMD}"
echo "Import time 1: $((${SECONDS}-${T2})) seconds"
echo "Performing second import of torch${IMPORT_INFO}"
T2=${SECONDS}
echo "${CMD}"
eval "${CMD}"
echo "Import time 2: $((${SECONDS}-${T2})) seconds"

# Define options to be passed to application.
# Exclamation mark used to avoid forced exit (with set -e)
# when read reaches end of stream (non-zero return code).
! read -r -d "" PYTHON_OPTS << EOS
 --ntasks-per-node ${SLURM_NTASKS_PER_NODE}\
 --dist-url ${DIST_URL}\
 --dist-port $(( (SLURM_JOB_ID % 10000) + 50000 ))\
 --cpus-per-task ${SLURM_CPUS_PER_TASK}\
 --epochs 400\
 --checkpoint-in no_checkpoint.pt\
 --checkpoint-out checkpoint400.pt\
 --checkpoint-interval 1\
 --log-interval 1\
 --lr 0.0002\
 --time-steps 500\
 --batch-size 32\
 --log-ranks\
 --cancel-on-exception
EOS

# Ensure that data needed are downloaded before running application.
echo ""
echo "Downloading/checking dataset"
T3=${SECONDS}
CMD="python -c \"import torchvision as tv; tv.datasets.MNIST('../data', download=True)\""
echo "${CMD}"
eval ${CMD}
echo ""
echo "Time downloading/checking dataset: $((${SECONDS}-${T3})) seconds"

# Run and time application.
T4=${SECONDS}
CMD="${LAUNCH} ${PYTHON_OPTS}"
echo ""
echo "PyTorch DDP run started: $(date)"
echo "${CMD}"
echo ""
${CMD}
echo ""
echo "PyTorch DDP run completed: $(date)"
echo "Run time: $((${SECONDS}-${T4})) seconds"
echo ""
echo "Job time: $((${SECONDS}-${T1})) seconds"
