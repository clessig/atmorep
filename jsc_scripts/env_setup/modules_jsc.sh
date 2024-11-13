#!/bin/bash
ml --force purge
ml use $OTHERSTAGES
ml Stages/2024

ml CUDA 

ml GCC/12.3.0
ml GCCcore/.12.3.0

ml OpenMPI/4.1.5

ml SciPy-bundle/2023.07
ml matplotlib/3.7.2
ml xarray/2023.8.0
ml dask/2023.9.2
ml ecCodes/2.31.0
ml zarr/2.18.3
ml PyYAML/6.0
ml netcdf4-python/1.6.4-serial
ml git/2.41.0-nodocs
