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
    packages=find_packages(),   
    # if packages are available in a native form fo the host system then these should be used
    install_requires=['torch==2.4', 'numpy', 'matplotlib', 'zarr', 'pandas', 'typing_extensions', 'pathlib', 'wandb', 'cloudpickle', 'ecmwflibs', 'cfgrib', 'netcdf4', 'xarray', 'pytz', 'torchinfo', 'pytest', 'dask'],
    data_files=[('./output', []), ('./logs', []), ('./results',[])],
)


#ATOS: 
# path = '/ec/res4/scratch/nacl/atmorep/'
#JSC : 
path = '/p/scratch/atmo-rep/data/era5_1deg/months/'
#BSC : 
#path = '/gpfs/scratch/ehpc03/'
assert os.path.exists(path), "The chosen data path does not exist on this device. Please change it in setup.py"

if not os.path.exists('./data'):
  os.system(f'ln -s {path} ./data')

if not os.path.exists('./output'):
  os.mkdir('./output')
if not os.path.exists('./logs'):
  os.mkdir('./logs')
if not os.path.exists('./results'):
  os.mkdir('./results')

