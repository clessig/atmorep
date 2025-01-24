import os 
from pathlib import Path

fpath = os.path.dirname(os.path.realpath(__file__))

path_models  = Path( fpath, '../../models/')
path_results = Path( fpath, '../../results')
path_plots   = Path( fpath, '../results/plots/') 

#link the following path to be the default data folder using the follwing command:
#ln -s <path> data 
#ATOS: path = /ec/res4/scratch/nacl/atmorep/
#JSC : path = /p/scratch/atmo-rep/data/era5_1deg/months/
#BSC : path = /gpfs/scratch/ehpc03/
path_data    = ('./data/era5_y2010_2021_res025_chunk8.zarr/')