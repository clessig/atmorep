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
    install_requires=['torch', 'numpy', 'matplotlib', 'zarr', 'pandas', 'typing_extensions', 'pathlib', 'wandb', 'cloudpickle', 'ecmwflibs', 'cfgrib', 'netcdf4', 'xarray', 'pytz', 'torchinfo', 'pytest', 'cfgrib'],
>>>>>>> 3868ab41a354c82de5a5ac7c71877c46f8016278
    data_files=[('./output', []), ('./logs', []), ('./results',[])],
)

if not os.path.exists('./output'):
  os.mkdir('./output')
if not os.path.exists('./logs'):
  os.mkdir('./logs')
if not os.path.exists('./results'):
  os.mkdir('./results')

