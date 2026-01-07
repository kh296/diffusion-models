#!/bin/bash
#SBATCH --job-name=diffusion-models_install  # create a name for your job
#SBATCH --output=%x.log         # job output file
#SBATCH --partition=pvc9        # cluster partition to be used
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1            # number of allocated gpus per node
#SBATCH --time=04:00:00         # total run time limit (HH:MM:SS)

T0=${SECONDS}
echo "Job start on $(hostname): $(date)"
echo ""

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

# Perform Dawn-specific setup.
if [[ "Dawn" == "${SYSTEM}" ]]; then
    module purge
    module load rhel9/default-dawn
fi

# Initialise conda.
if [ -z "${CONDA_HOME}" ]; then
    CONDA_HOME="$(realpath ~)/miniforge3"
fi
source ${CONDA_HOME}/bin/activate

# Delete any pre-existing environment.
if [ -d "${CONDA_HOME}/envs/${ENV_NAME}" ]; then
    conda env remove -n ${ENV_NAME} -y
fi

# Create and activate the environment.
conda create -n ${ENV_NAME} -y python=3.13
conda activate ${ENV_NAME}

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

echo ""
echo "Job completion on $(hostname): $(date)"
echo "Time to complete job: $((${SECONDS}-${T0})) seconds"
