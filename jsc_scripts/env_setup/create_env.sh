#!/usr/bin/env bash
#
# __authors__ = Michael Langguth
# __date__  = '2022-01-21'
# __update__= '2024-11-12'
#
# **************** Description ****************
# This script can be used for setting up the virtual environment needed for AtmoRep.
# **************** Description ****************
#
### MAIN S ###
#set -eu              # enforce abortion if a command is not re

SCR_SETUP="%create_env.sh: "

## some first sanity checks
# script is sourced?
if [[ ${BASH_SOURCE[0]} == "${0}" ]]; then
  echo "${SCR_SETUP}ERROR: 'create_env.sh' must be sourced, i.e. execute by prompting 'source create_env.sh [virt_env_name]'"
  exit 1
fi

# from now on, just return if something unexpected occurs instead of exiting
# as the latter would close the terminal including logging out
if [[ -z "$1" ]]; then
  echo "${SCR_SETUP}ERROR: Provide a name to set up the virtual environment, i.e. execute by prompting 'source create_env.sh [virt_env_name]"
  return
fi

# set some variables
HOST_NAME=$(hostname)
ENV_NAME=$1
SETUP_DIR=$(pwd)
SETUP_DIR_NAME="$(basename "${SETUP_DIR}")"
BASE_DIR="$(dirname "${SETUP_DIR}")"
# set-up directory for virtual environment
if [[ -z "$2" ]]; then
    VENV_BASE_DIR="${BASE_DIR}/virtual_envs/"
else
    VENV_BASE_DIR="$2/virtual_envs/"
fi
echo "${SCR_SETUP}Virtual environemnt will be set up under ${VENV_BASE_DIR}..."
VENV_DIR="${VENV_BASE_DIR}/${ENV_NAME}"
ATMOREP_DIR="$(dirname "${BASE_DIR}")"

## perform sanity checks
# * check if script is called from env_setup-directory
# * check if virtual env has already been set up

# script is called from env_setup-directory?
if [[ "${SETUP_DIR_NAME}" != "env_setup"  ]]; then
  echo "${SCR_SETUP}ERROR: Execute 'create_env.sh' from the env_setup-subdirectory only!"
  echo "${SETUP_DIR_NAME}"
  return
fi

# virtual environment already set-up?
if [[ -d ${VENV_DIR} ]]; then
  echo "${SCR_SETUP}Virtual environment has already been set up under ${VENV_DIR} and is ready to use."
  echo "NOTE: If you wish to set up a new virtual environment, delete the existing one or provide a different name."
  ENV_EXIST=1
else
  ENV_EXIST=0
fi

## check integratability of operating system
if [[ "${HOST_NAME}" == *jrlogin* || "${HOST_NAME}" == *jwlogin* || "${HOST_NAME}" == *jrc* ]]; then
  # unset PYTHONPATH to ensure that system-realted paths are not set
  unset PYTHONPATH
  modules_file="modules_jsc.sh"
else
  echo "${SCR_SETUP}ERROR: Software stack can only be set-up for JURECA and Juwels (Booster) so far."
  return
fi

## set up virtual environment

ACT_VIRT_ENV=${VENV_DIR}/bin/activate

if [[ "$ENV_EXIST" == 0 ]]; then
  # Install virtualenv-package and set-up virtual environment with required additional Python packages.
  echo "${SCR_SETUP}Configuring and activating virtual environment on ${HOST_NAME}"

  source "${modules_file}"

  python3 -m venv --system-site-packages "${VENV_DIR}"

  echo "${SCR_SETUP}Entering virtual environment ${VENV_DIR} to install required Python modules..."
  source "${ACT_VIRT_ENV}"
 
  # handle systematic issues with Stages/2022 
  MACHINE=$(hostname -f | cut -d. -f2)
  if [[ "${HOST}" == jwlogin2[2-4] ]]; then
     MACHINE="juwelsbooster"
  fi
  PY_VERSION=$(python --version 2>&1 | cut -d ' ' -f2 | cut -d. -f1-2)

  echo "${SCR_SETUP}Appending PYTHONPATH on ${MACHINE} for Python version ${PY_VERSION} to ensure proper set-up..."

  req_file=${SETUP_DIR}/requirements.txt

  # Install additional requirements
  pip3 install --no-cache-dir -r "${req_file}"
  # Install atmorep and its dependencies
  pip3 install -e ../../

  # expand PYTHONPATH
  ##export PYTHONPATH=${ATMOREP_DIR}:$PYTHONPATH >> "${activate_virt_env}"

  ## ...and ensure that this also done when the
  #echo "" >> "${activate_virt_env}"
  #echo "# Expand PYTHONPATH..." >> "${activate_virt_env}"
  #echo "export PYTHONPATH=${ATMOREP_DIR}:\$PYTHONPATH" >> "${activate_virt_env}"

  # deactivate virtual environment
  # NOTE: virtual environment must be deactivated when submitting batch-scripts to avoid issues
  deactivate

  info_str="Virtual environment ${VENV_DIR} has been set up successfully."
elif [[ "$ENV_EXIST" == 1 ]]; then
  # simply activate virtual environment
  info_str="Virtual environment ${VENV_DIR} has already been set up before. Nothing to be done."
  
  source "${activate_virt_env}"
fi

echo "${SCR_SETUP}${info_str}"
### MAIN E ###
