import os 
from pathlib import Path

fpath = os.path.dirname(os.path.realpath(__file__))

year_base = 1979
year_last = 2021

path_models = Path( fpath, '../../models/')
path_results = Path( fpath, '../../results/')
path_data = Path( fpath, '../../data/')
path_plots = Path( fpath, '../results/plots/')

grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
                'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                'total_precip' : 'tp', 'radar_precip' : 'yw_hourly',
                't2m' : 't_2m', 'u_10m' : 'u_10m', 'v_10m' : 'v_10m',  }
