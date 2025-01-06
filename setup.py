from setuptools import setup, find_packages
import sys
import os

setup(
    name='atmorep',
    version='0.1',
    url='https://github.com/clessig/atmorep',
    author='Christian Lessig',
    author_email='christian@atmorep.org',
    description='AtmoRep',
    python_requires=">=3.11",
    packages=find_packages(),   
    # if packages are available in a native form fo the host system then these should be used
    install_requires=['torch>=2.3', 'numpy', 'matplotlib', 'zarr', 'pandas', 'typing_extensions', 'pathlib', 'wandb', 'cloudpickle', 'ecmwflibs', 'cfgrib', 'netcdf4', 'xarray', 'pytz', 'torchinfo', 'pytest', 'dask'],
)
