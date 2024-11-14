#!/usr/bin/python3

import platform
import subprocess
import pathlib
import os

DRY_RUN = True

ATMOREP_BASE_DIR = pathlib.Path(__file__).parent
SCRIPTS_DIR =  ATMOREP_BASE_DIR / "jsc_scripts"
OUTPUT_DIR = ATMOREP_BASE_DIR / "output"
LOG_DIR = ATMOREP_BASE_DIR / "logs"

SUBMISSION_SCRIPT = "slurm_atmorep_evaluate.sh"
LOG_NAME = r"atmorep-%x.%j"

JSC_DOMAINS = {"juwels", "jureca"}
SUBMISSION_COMMAND = "sbatch"

def main():
    short_name, domain = platform.node().split(".")
    # assure that $SLURM_SUBMIT_DIR == ATMOREP_BASE_DIR
    os.chdir(ATMOREP_BASE_DIR)
    
    args = []
    if domain in JSC_DOMAINS:
        args = [
            SUBMISSION_COMMAND,
            SCRIPTS_DIR / SUBMISSION_SCRIPT,
            f"--output={LOG_DIR/LOG_NAME}.out",
            f"--error={LOG_DIR/LOG_NAME}.err"
        ]
    
    print("running command:")
    print(" ".join((str(arg) for arg in args)))
    if not DRY_RUN:    
        subprocess.run(args)
        

def setup_parser():
    pass

if __name__ == "__main__":
    main()