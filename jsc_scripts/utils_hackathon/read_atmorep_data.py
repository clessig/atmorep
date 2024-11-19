
# SPDX-FileCopyrightText: 2024 AtmoRep collaboration: European Centre for Medium-Range Forecasting (ECMWF), Jülich Supercomputing Center (JSC), European Center for Nuclear Research (CERN)
#
# SPDX-License-Identifier: MIT

"""
Class to read in AtmoRep output from zarr store. 
Date: July 2023
Update: 2024-11-19
By: Michael Langguth (JSC)
"""

# import packages
import os
import json
from typing import List
#from tqdm import tqdm
import zarr
import numpy as np
import xarray as xr
from pathlib import Path

class HandleAtmoRepData(object):
    """
    Handle outout data of AtmoRep.
    TODO:
    - support fixed location sampling
    """
    known_data_types = ["source", "pred", "target", "ens"]
    
    def __init__(self, model_id: str, results_basedir: str = "/p/scratch/atmo-rep/results/",):
        """
        :param model_id: ID of Atmorep-run to process
        :param results_basedir: top-directory where results are stored
        """
        self.model_id = model_id if model_id.startswith("id") else f"id{model_id}"
        self.results_dir = results_basedir
        self.config_file, self.config = self._get_config()
        
        self.target_type = "fields_prediction" if self.config["BERT_strategy"] in ["forecast", "BERT"] else "fields_targets"
        
        self.input_variables = self._get_invars()
        self.target_variables = self._get_tarvars()
        
        self.input_token_config = self.get_input_token_config()
        self.target_token_config = self.get_target_token_config() 
        
    @property
    def results_dir(self):
        return self._results_dir
    
    @results_dir.setter
    def results_dir(self, results_basedir):
        results_dir = Path(os.path.join(results_basedir, self.model_id))
        config_file =  results_dir.joinpath(f"model_{self.model_id}.json")
        zarr_files = list(results_dir.glob('*.zarr'))
        
        # basic checks: directory existence, availability of config-file and zarr-files
        if not results_dir.is_dir():
            raise NotADirectoryError(f"Could not find AtmoRep results-directory '{results_dir}'")
        
        if not config_file.exists():
            raise FileNotFoundError(f"Results directory exist, but configuration file '{config_file}' cannot be found.")
            
        if len(zarr_files) == 0:
            raise FileNotFoundError(f"Results directory exist, but no zarr-output files found.")
            
        self._results_dir = results_dir

        
    def _get_config(self) -> (str, dict):
        """
        Get configuration dictionary of trained AtmoRep-model.
        """
        config_jsf = self.results_dir.joinpath(f"model_{self.model_id}.json")  #os.path.join(self.datadir, f"model_{self.model_id}.json")
        with open(config_jsf) as json_file:
            config = json.load(json_file)
        return config_jsf, config
    
    def _get_invars(self) -> List:
        """
        Get list of input variables of trained AtmoRep-model.
        """
        return [var_list[0] for var_list in self.config["fields"]]
    
    def _get_tarvars(self) -> List:
        """
        Get list of target variables of trained AtmoRep-model.
        """
        return [var_list[0] for var_list in self.config[self.target_type]]
    
    
    def _get_token_config(self, key) -> dict:
        """
        Generic function to retrieve token configuration.
        :param key: key-string from which token-info is deducable
        :return dictionary of token info
        """
        token_config_keys = ["general_config", "vlevel", "num_tokens", "token_shape", "bert_parameters"]
        token_config = {var[0]: None for var in self.config[key]}        
        for i, var in enumerate(token_config):
            len_config = len(self.config[key][i])
            token_config[var] =  {config_key: self.config[key][i][j+1] for j, config_key in enumerate(token_config_keys)}
        
        return token_config
    
    def get_input_token_config(self) -> dict:
        """
        Get input token configuration
        """
        return self._get_token_config("fields")
        
    def get_target_token_config(self) -> dict:
        """
        Retrieve token configuration of output/target data.
        Note that the token configuration is the same as the input data as long as target_fields is unset.
        """
        if self.target_type in ["target_fields", "fields_targets"]:
            return self._get_token_config(self.target_type)   
        else:
            return self.input_token_config
    
    
    def read_one_structured_file(self, fname: str, varname: str, data_type: str):
        """
        Read data from a single, structured output file of AtmoRep and convert to xarray DataArray with underlying coordinate information.
        Here, structured means that the output is not randomly masked (i.e. BERT) nor has an (implicit) lead-time dimension 
        :param fname: path to zarr-store that should be read
        :param varname: name of variable in zarr-store to be read
        :param data_type: Type of data which should be retrieved (either 'source', 'target', 'ens' or 'pred')
        :return: list of DataArrays where each element provides data from a local neighborhood tile; 
                 dimensions (["ensemble"], ml", "datetime", "lat", "lon")
        """    
        store = zarr.ZipStore(fname)
        grouped_store = zarr.group(store)
            
        dims = ["ml", "datetime", "lat", "lon"]
        if data_type == "ens":
            nens = self.config["net_tail_num_nets"]
            coords = {"ensemble": range(nens)}
        else:
            coords = {}
            
        da = []
        for _, patch in enumerate(grouped_store[os.path.join(varname)]):    
            coords.update({dim: grouped_store[os.path.join(varname, patch, dim)] for dim in dims})
            da_p = xr.DataArray(grouped_store[os.path.join(varname, patch, "data")], coords=coords,                
                                dims = ["ensemble"] + dims if data_type == "ens" else dims, name=f"{varname}_{patch.replace('=', '')}")            
            
            da.append(da_p)
        
        # ML: This would trigger loading data into memory rather than lazy data access.
        #da = xr.concat(da, "patch")
        
        return da

    def read_one_forecast_file(self, fname: str, varname: str, data_type: str):
        """
        Read data from a single, forecast output file of AtmoRep and convert to xarray DataArray with underlying coordinate information.
        :param fname: path to zarr-store that should be read
        :param varname: name of variable in zarr-store to be read
        :param data_type: type of data which should be retrieved (either 'source', 'target', 'ens' or 'pred')
        :return: list of DataArrays where each element provides data from a local neighborhood tile; 
                 dimensions (["ensemble"], ml", "init_time", "lead_time", "lat", "lon")
        """    
        store = zarr.ZipStore(fname)
        grouped_store = zarr.group(store)
            
        dims = ["ml", "datetime", "lat", "lon"]
        idx_init = 1                         # index of axis to insert init_time-dimension
        if data_type == "ens":
            nens = self.config["net_tail_num_nets"]
            coords = {"ensemble": range(nens)}
            idx_init = 2
        else:
            coords = {}


        # get lead time array from config
        nfcst_tokens = self.config["forecast_num_tokens"]
        vars = np.array([self.config["fields"][i][0] for i in range(len(self.config["fields"]))])
        ivar = np.where(varname == vars)[0][0]
    
        tsteps_token = self.config["fields"][ivar][4][0]
        lead_time = range(1, nfcst_tokens*tsteps_token + 1)
    
        da = []
        for _, patch in enumerate(grouped_store[os.path.join(varname)]):    
            coords.update({dim: grouped_store[os.path.join(varname, patch, dim)] for dim in dims})
            da_p = xr.DataArray(grouped_store[os.path.join(varname, patch, "data")], coords=coords,                
                                dims = ["ensemble"] + dims if data_type == "ens" else dims, name=f"{varname}_{patch.replace('=', '')}")
            # split datetime-dimension to (init_time, lead_time)
            da_p = da_p.assign_coords({"lead_time": ("datetime", lead_time)})\
                       .expand_dims({"init_time": [coords["datetime"][0] - np.timedelta64(1, "h")]}, idx_init)\
                       .swap_dims({"datetime": "lead_time"}).drop("datetime")  

            da.append(da_p)
        
        
        return da
    
    def read_one_bert_file(self, fname: str, varname: str, data_type: str, ml: int):
        """
        Read data from single AtmoRep output file that has been written in BERT-mode
        :param fname: path to zarr-store that should be read
        :param varname: name of variable in zarr-store to be read
        :param data_type: type of data which should be retrieved (either 'source', 'target', 'ens' or 'pred')
        :param ml: model-level for which data should be read 
        :return: list of xarray.DataArrays with dimensions ("itoken", "t", "y", "x")
        """        
        store = zarr.ZipStore(fname)
        grouped_store = zarr.group(store)
        
        dims = ["itoken", "t", "y", "x"]
        dims_map = {"datetime": ("itoken", "t"), "lat": ("itoken", "y"), "lon": ("itoken", "x")}
        if data_type == "ens":
            dims.insert(1, "ensemble") 
            
        da = []
            
        for _, patch in enumerate(grouped_store[os.path.join(varname)]):
            data_coords = {dim: (dim_map, grouped_store[os.path.join(varname, patch, f"ml={ml:05d}", dim)]) for dim, dim_map in dims_map.items()}
            data = grouped_store[os.path.join(varname, patch, f"ml={ml:05d}", "data")]
            data_sh = data.shape
            _ = [data_coords.update({dim: range(data_sh[i]) for i, dim in enumerate(dims)})]

            da_p = xr.DataArray(grouped_store[os.path.join(varname, patch, f"ml={ml:05d}", "data")], 
                                coords=data_coords, dims = dims, name=f"{varname}_ml{ml:05d}_{patch.replace('=', '')}")
            da.append(da_p)

        return da
    
    def read_data(self, varname: str, data_type, epoch: int = -1, **kwargs):
        """
        Read data from a single output file of AtmoRep and convert to xarray DataArray with underlying coordinate information.
        :param varname: name of variable for which token info is requested
        :param data_type: Type of data which should be retrieved (either 'source', 'target', 'ens' or 'pred')
        :param epoch: training epoch of requested token information file
        """                  
        assert data_type in self.known_data_types, f"Data type '{data_type}' is unknown. Chosse one of the following: '{', '.join(self.known_data_types)}'"
        
        filelist = self.get_hierarchical_sorted_files(data_type, epoch)
        nfiles = len(filelist)

        args = {"varname": varname, "data_type": data_type}
        if self.config["BERT_strategy"] in ["forecast", "global_forecast"]:
            self.read_one_file = self.read_one_forecast_file
        elif self.config["BERT_strategy"] == "BERT":
        #elif self.config["BERT_strategy"] == "BERT" or self.config["BERT_strategy"] == "temporal_interpolation":
            assert isinstance(kwargs.get("ml", None), int), f"Model-level ml must be an integer, but '{kwargs.get('ml', None)}' was parsed."
            self.read_one_file = self.read_one_bert_file
            # source data is still structured
            if data_type == "source":
                self.read_one_file = self.read_one_structured_file
            else:
                # Add ml-key to args-dictionary for reading-method
                args = {"varname": varname, "ml": kwargs.get("ml"), "data_type": data_type}
            
        else:
            self.read_one_file = self.read_one_structured_file
            #print(f"Handling data with sampling strategy '{self.config['BERT_strategy']}' is not supported yet.")
        msg = f"Start reading {nfiles} zarr stores..." if nfiles > 1 else f"Start reading one zarr store..." 
        print(msg)
        
        da = []
        for _, f in enumerate(filelist):
            da += self.read_one_file(f, **args)
            
        # return global data if global forecasting evaluation mode was chosen 
        if self.config["BERT_strategy"] == "global_forecast":
            da = self.get_global_field(da)

        return da
    
    @staticmethod
    def get_global_field(da_list):
        
        # get resolution from data
        all_init_times = list(set([time for da in da_list for time in da["init_time"].values]))
        dx, dy = np.abs(da_list[0]["lon"][1] - da_list[0]["lon"][0]), \
                 np.abs(da_list[0]["lat"][1] - da_list[0]["lat"][0])
        
        # initialize empty global data array
        dims = da_list[0].dims
        data_coords = {dim: da_list[0].coords[dim] for dim in dims if dim not in ["lat", "lon"]}
        data_coords["lat"] = np.linspace(-90., 90., num=int(180/dy) + 1, endpoint=True)
        data_coords["lon"] = np.linspace(0, 360, num=int(360/dx), endpoint=False)  
        data_coords["init_time"] = all_init_times

        da_global = xr.DataArray(np.empty(tuple(len(d) for d in data_coords.values())), 
                                 coords=data_coords, dims=dims)
        # fill global data array 
        for da in da_list:
            da_global.loc[{"init_time": da["init_time"], "lat": da["lat"], "lon": da["lon"]}] = da

        if np.any(da_global.isnull()): 
            raise ValueError(f"Could not get global data field.")
            
        return da_global                       
    
        
    def get_hierarchical_sorted_files(self, data_type: str, epoch: int = -1):
        """
        Get sorted list of file names.
        :param data_type: Type of data which should be retrieved (either 'source', 'target', 'ens' or 'pred')
        :param epoch: number of epoch for which data should be retrieved. Parse -1 for getting all epochs.
        """
        epoch_str = f"epoch*" if epoch == -1 else f"epoch{epoch:05d}"
        
        fpatt = f"results_{self.model_id}_{epoch_str}_{data_type}.zarr"
        filelist = list(Path(self.results_dir).glob(f"**/{fpatt}"))
        
        if len(filelist) == 0:
            raise FileNotFoundError(f"Could not file any files mathcing pattern '{fpatt}' under directory '{self.results_dir}'.")
        
        # hierarchical sorting: epoch -> rank -> batch
        sorted_filelist = sorted(filelist, key=lambda x: self.get_number(x, "_epoch"))
      
        return sorted_filelist
    
    @staticmethod
    def get_number(file_name, split_arg):
        # Extract the number from the file name using the provided split argument
        return int(str(file_name).split(split_arg)[1].split('_')[0])  

    
