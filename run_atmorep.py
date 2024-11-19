#!/usr/bin/python3

import platform
import subprocess
import pathlib
import os
from argparse import ArgumentParser

DRY_RUN = False

ATMOREP_BASE_DIR = pathlib.Path(__file__).parent
SCRIPTS_DIR =  ATMOREP_BASE_DIR / "jsc_scripts"
OUTPUT_DIR = ATMOREP_BASE_DIR / "output"
LOG_DIR = ATMOREP_BASE_DIR / "logs"

SUBMISSION_SCRIPT = {
    "evaluate": "slurm_atmorep_evaluate.sh",
    "train": "slurm_atmorep_train.sh",
    "train_multi": "slurm_atmorep_train_multi.sh",
}

LOG_NAME = r"atmorep-%x.%j"

JSC_DOMAINS = {"juwels", "jureca"}
SUBMISSION_COMMAND = "sbatch"

def main(submission_type: str):
    general_args = [
        f"--output={LOG_DIR/LOG_NAME}.out",
        f"--error={LOG_DIR/LOG_NAME}.err",
        f"--time=0-0:20:00",
        #f"--nodes=1"
    ]
    
    submission_script = SUBMISSION_SCRIPT[submission_type]
    prepare_job()
    submit_job(general_args, submission_script)
    
def get_platform():
    return platform.node().split(".")

def prepare_job():
    # assure that $SLURM_SUBMIT_DIR == ATMOREP_BASE_DIR
    os.chdir(ATMOREP_BASE_DIR)
    
    LOG_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True) #TODO: handle with rest of user config

def submit_job(general_args, submission_script):
    args = [SUBMISSION_COMMAND, *general_args, SCRIPTS_DIR / submission_script]
    print("running command:")
    print(" ".join((str(arg) for arg in args)))
    if not DRY_RUN:    
        subprocess.run(args)
    else:
        print("dry run")
        
if __name__ == "__main__":
    parser = ArgumentParser(
        description="Wrapper to easily configure/run slurm submissions."
    )
    parser.add_argument(
        "submission_type",
        choices=SUBMISSION_SCRIPT.keys(),
        help="Choose what kind of submission to run."
    )
    
    args = parser.parse_args()
    
    main(args.submission_type)