import os 
from pathlib import Path

fpath = os.path.dirname(os.path.realpath(__file__))

year_base = 1979
year_last = 2022

path_models = Path( fpath, '../../models/')
path_results = Path( fpath, '../../results')
path_data = Path( fpath, '../../data/')
path_plots = Path( fpath, '../results/plots/')

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
                'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 'radar_precip' : 'yw_hourly',
                't2m' : 't_2m', 'u_10m' : 'u_10m', 'v_10m' : 'v_10m',  }

# TODO: extract this info from the datasets
datasets = {}
#
datasets['era5'] = {}
datasets['era5']['resolution'] = [1, 0.25, 0.25]
datasets['era5']['extent'] = [ [1979, 2022], [90., -90], [0.0, 360] ]
datasets['era5']['is_global'] = True
datasets['era5']['file_size'] = [ -1, 721, 1440]
#
datasets['cosmo_rea6'] = {}
datasets['cosmo_rea6']['resolution'] = [1, 0.0625, 0.0625]
datasets['cosmo_rea6']['extent'] = [ [1997, 2017], [27.5,70.25], [-12.5,37.0] ]
datasets['cosmo_rea6']['is_global'] = False
datasets['cosmo_rea6']['file_size'] = [ -1, 685, 793]
#
datasets['cerra'] = {}
datasets['cerra']['resolution'] = [3, 0.25, 0.25]
datasets['cerra']['extent'] = [ [1985, 2001], [75.25,20.5], [-58.0,74.0] ]
datasets['cerra']['is_global'] = False
datasets['cerra']['file_size'] = [ -1, 220, 529]
