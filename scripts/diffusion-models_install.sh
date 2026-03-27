#!/bin/bash
#SBATCH --job-name=diffusion-models_install  # create a name for your job
#SBATCH --output=%x.log         # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of allocated gpus per node
#SBATCH --time=04:00:00         # total run time limit (HH:MM:SS)

# Script for installing packages for running on the Dawn supercomputer,
# and use on the Dawn supercomputer, and on other systems, (most of)
# the Accelerate Science examples for diffusion models:
# https://github.com/kh296/diffusion-models/tree/xpu
# This includes installation of pytorch (version 2.9.1),
#
# This installation relies on the user having a conda installation
# at ${CONDA_HOME}/bin/activate.  If not set by the user, CONDA_HOME
# defaults to ${HOME}/miniforge3.  For instructions for installing
# the Miniforge3 flavour of conda, see: https://conda-forge.org/download/
#
# After installation, the environment for using the diffusion-model
# packages be activated by sourcing the file diffusion-models-setup.sh,
# created in the directory ../envs relative to where the current script is run.
#
# On Dawn, the current script may be run interactively on a compute node
# (not on a login node):
# bash ./diffusion-models_install.sh
# or it may be submitted from a login node to the Slurm batch system:
# sbatch --account=<project account> ./diffusion-models_install.sh
#

# Exit at first failure.
set -e

ENV_NAME="diffusion-models"

# Determine system being used.
if [[ "$(hostname)" == "pvc-s"* ]]; then
    SYSTEM="Dawn"
elif [[ "${OSTYPE}" == "darwin"* ]]; then
    SYSTEM="macOS"
else
    echo "Installation of ${ENV_NAME} for ${OSTYPE} on $(hostname) not handled"
    echo "Exiting: $(date)"
    exit 1
fi
#
# Check that conda is available.
if [ -z "${CONDA_HOME}" ]; then
    if [ -z "${CONDA_PREFIX}" ]; then
        CONDA_HOME=${HOME}/miniforge3
    else
        CONDA_HOME=${CONDA_PREFIX}
    fi
fi

if ! [ -d "${CONDA_HOME}" ]; then
    echo "Conda installation not found at ${CONDA_HOME}"
    echo "Exiting: $(date)"
    exit 2
fi
#
# Perform installation.
echo "Installation of ${ENV_NAME} for ${OSTYPE} on $(hostname) started: $(date)"
T0=${SECONDS}

# Create script for environment setup.
ENVS_DIR=$(realpath ..)/envs
mkdir -p ${ENVS_DIR}
SETUP="${ENVS_DIR}/${ENV_NAME}-setup.sh"
DAWN_SETUP="/dev/null"
MACOS_SETUP="/dev/null"
if [[ "Dawn" == "${SYSTEM}" ]]; then
    DAWN_SETUP="${SETUP}"
elif [[ "macOS" == "${SYSTEM}" ]]; then
    MACOS_SETUP="${SETUP}"
fi

cat <<EOF >${SETUP}
# Setup script for ${ENV_NAME} on ${SYSTEM}.
# Generated: $(date)

EOF

cat <<EOF >>${DAWN_SETUP}
# Load modules.
module purge
module load rhel9/default-dawn
module load intel-oneapi-ccl/2021.15.0

#
# Set level-zero environment variables:
# https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html#environment-variables
#

# Define device hierarchy model and affinity mask.
# See: https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
# Define whether a GPU is treated as a single root device ("COMPOSITE")
# or as a root device per stack ("FLAT").
export ZE_FLAT_DEVICE_HIERARCHY="FLAT"
# Define root devices per node to be made visible to applications.
# (Dawn has 4 GPUs per node, and two stacks per GPU.)
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7"

#
# Set some variables relevant to Intel MPI Library:
# https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-15/environment-variable-reference.html
#

# Set variables relating to GPU support.
# See: https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-15/gpu-support.html
# Disable/enable GPU support (default: 0).
export I_MPI_OFFLOAD=1
# Disable/enable GPU pinning (default: 0).
export I_MPI_OFFLOAD_PIN=1
# Enable/disable assumption that all buffers in an operation have the same type
# (default: 0).
export I_MPI_OFFLOAD_SYMMETRIC=0

# Set hydra environment variables.
# See: https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-15/hydra-environment-variables.html
# Disable/enable process placement provided by job scheduler (default:1)
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0
# Set bootstrap server (default:"ssh")
export I_MPI_HYDRA_BOOTSTRAP="slurm"

# Configure debug output.
# See: https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-15/other-environment-variables.html
# See: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-0/intel-mpi-for-gpu-clusters.html
export I_MPI_DEBUG=0

#
# Set some variables relevant to OneAPI collective communications library
# (oneCCL):
# https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-ze-ipc-exchange
#

# Select transport for inter-process communication (default: "mpi").
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-atl-transport
export CCL_ATL_TRANSPORT="ofi"

# Set CCL log level (default: "warn").
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-log-level
export CCL_LOG_LEVEL="warn"

# Set CCL process launcher (default: "hydra).
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-process-launcher
export CCL_PROCESS_LAUNCHER="hydra"

# Set mechanism for CCL level zero inter-process communications
# (default: pidfd).
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-ze-ipc-exchange
export CCL_ZE_IPC_EXCHANGE=sockets

# Define filters for selection multiple network interfaces cards (NICs).
# See:
# https://uxlfoundation.github.io/oneCCL/env-variables.html#multi-nic
#
# Control multi-NIC selection by NIC locality.
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-ze-ipc-exchange
# export CCL_MNIC="none"
#
# Control multi-NIC selection by NIC names.
# See: https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-mnic-name
#export CCL_MNIC_NAME=
#
# Specify the maximum number of NICs to be selected.
# https://uxlfoundation.github.io/oneCCL/env-variables.html#ccl-mnic-count
#export CCL_MNIC_COUNT=


# Avoid CCL warning:
# [CCL_WARN] CCL_CONFIGURATION_PATH_modshare=:1 is unknown to and unused by
# oneCCL code but is present in the environment, check if it is not mistyped.
unset CCL_CONFIGURATION_PATH_modshare

EOF

cat <<EOF >>${MACOS_SETUP}
# Initialise environment variables that may be used at run time.
# Define network interface.
export GLOO_SOCKET_IFNAME="en0"

EOF

cat <<EOF >>${SETUP}
# Initialise conda.
source $(realpath ${CONDA_HOME})/bin/activate

# Activate environment.
EOF

# Set up installation environment.
source ${SETUP}
conda update -n base -c conda-forge conda -y

# Delete any pre-existing environment.
if [ -d "${CONDA_HOME}/envs/${ENV_NAME}" ]; then
    conda env remove -n ${ENV_NAME} -y
fi

# Create and activate the environment.
conda create -n ${ENV_NAME} -y python=3.13
CMD="conda activate ${ENV_NAME}"
echo "${CMD}" >> "${SETUP}"
eval "${CMD}"

# Install additional packages.
python -m pip install --upgrade pip
if [[ "Dawn" == "${SYSTEM}" ]]; then
    OPTS="--index-url https://download.pytorch.org/whl/xpu"
else
    OPTS=""
fi
# Includes constraints on fsspec version of datasets packages.
pip install ${OPTS} fsspec\<=2025.10.0 torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1
pip install datasets diffusers ipykernel ipywidgets jupyterlab matplotlib scikit-learn seaborn transformers

# Check installation by importing modules.
CMD="python -c 'import datasets; import diffusers; import ipykernel; import ipywidgets; import jupyterlab; import matplotlib; import sklearn; import seaborn; import torch; import torchaudio; import torchvision; import transformers'"
echo ""
echo "Performing initial imports:"
echo "${CMD}"
eval "${CMD}"

# Create Jupyter kernel in default location.
CMD="python -m ipykernel install --user --name=${ENV_NAME}"
echo ""
echo "Creating Jupyter kernel:"
echo "${CMD}"
eval "${CMD}"

echo ""
echo "Installation of ${ENV_NAME} for ${OSTYPE} on $(hostname) completed: $(date)"
echo "Installation time: $((${SECONDS}-${T0})) seconds"
