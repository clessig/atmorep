####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import torch
import pathlib
import numpy as np
import xarray as xr
from functools import partial

import atmorep.utils.utils as utils
from atmorep.config.config import year_base
from atmorep.utils.utils import tokenize
from atmorep.datasets.file_io import grib_file_loader, netcdf_file_loader, bin_file_loader

# TODO, TODO, TODO: replace with torch functonality
# import cv2 as cv

class DataLoader:
    
    def __init__(self, path, file_shape, data_type = 'reanalysis',
                       file_format = 'grib', level_type = 'pl',
                       fname_base = '{}/{}/{}/{}{}/{}_{}_y{}_m{}_{}{}',
                       smoothing = 0,
                       log_transform = False,
                       partial_load = 0):

      self.path = path
      self.data_type = data_type
      self.file_format = file_format
      self.file_shape = file_shape
      self.fname_base = fname_base
      self.smoothing = smoothing
      self.log_transform = log_transform
      self.partial_load = partial_load

      if 'grib' == file_format :
        self.file_ext = '.grib'
        self.file_loader = grib_file_loader
      elif 'binary' == file_format :
        self.file_ext = '_fp32.dat'
        self.file_loader = bin_file_loader
      elif 'netcdf4' == file_format :
        self.file_ext = '.nc4'
        self.file_loader = netcdf_file_loader
      elif 'netcdf' == file_format :
        self.file_ext = '.nc'
        self.file_loader = netcdf_file_loader
      else :
        raise ValueError('Unsupported file format.')

      self.fname_base = fname_base + self.file_ext

      self.grib_index = { 'vorticity' : 'vo', 'divergence' : 'd', 'geopotential' : 'z',
                          'orography' : 'z', 'temperature': 't', 'specific_humidity' : 'q',
                          'mean_top_net_long_wave_radiation_flux' : 'mtnlwrf',
                          'velocity_u' : 'u', 'velocity_v': 'v', 'velocity_z' : 'w',
                          'total_precip' : 'tp', 'radar_precip' : 'yw_hourly',
                          't2m' : 't_2m', 'u_10m' : 'u_10m', 'v_10m' : 'v_10m',  }
      
    def get_field( self, year, month, field, level_type, vl, 
                   token_size = [-1, -1], t_pad = [-1, -1, 1]):
        
      t_srate = t_pad[2]
      data_ym = torch.zeros( (0, self.file_shape[1], self.file_shape[2]))

      # pre-fill fixed values
      fname_base = self.fname_base.format( self.path, self.data_type, field, level_type, vl,
                                           self.data_type, field, {},{},{},{})

      # padding pre
      if t_pad[0] > 0 :
          if month > 1:
              month_p = str(month-1).zfill(2)
              days_month = utils.days_in_month( year, month-1)
              fname = fname_base.format( year, month_p, level_type, vl)
          else:
              assert(year >= year_base)
              year_p = str(year-1).zfill(2)
              days_month = utils.days_in_month( year, 12)
              fname = fname_base.format( year-1, 12, level_type, vl)
          x = self.file_loader( fname, self.grib_index[field], [t_pad[0], 0, t_srate],
                                days_month )

          data_ym = torch.cat((data_ym,x),0)
    
      # data
      fname = fname_base.format( year, str(month).zfill(2), level_type, vl)
      days_month = utils.days_in_month( year, month)
      x = self.file_loader(fname,self.grib_index[field], [0, self.partial_load, t_srate],days_month)

      data_ym = torch.cat((data_ym,x),0)

      # padding post
      if t_pad[1] > 0 :
          if month > 1:
              month_p = str(month+1).zfill(2)
              days_month = utils.days_in_month( year, month+1)
              fname = fname_base.format( year, month_p, level_type, vl)
          else:
              assert(year >= year_base)
              year_p = str(year+1).zfill(2)
              days_month = utils.days_in_month( year+1, 12)
              fname = fname_base.format( year_p, 12, level_type, vl)
          x = self.file_loader( fname, self.grib_index[field], [0, t_pad[1], t_srate],
                                days_month)

          data_ym = torch.cat((data_ym,x),0)

      if self.smoothing > 0 :
          sm = self.smoothing
          mask_nan = torch.isnan( data_ym)
          data_ym[ mask_nan ] = 0.
          blur = partial( cv.blur, ksize=(sm,sm), borderType=cv.BORDER_REFLECT_101)
          data_ym = [torch.from_numpy( blur( data_ym[k].numpy()) ).unsqueeze(0) 
                          for k in range(data_ym.shape[0])]
          data_ym = torch.cat( data_ym, 0)
          data_ym[ mask_nan ] = torch.nan

      # tokenize
      data_ym = tokenize( data_ym, token_size)

      return data_ym

    def get_single_field( self, years_months, field = 'vorticity', level_type = 'pl', vl = 975, 
                          token_size = [-1, -1], t_pad = [-1, -1, 1]):
        
      data_field = []
      for year, month in years_months :
        data_field.append( self.get_field( year, month, field, level_type, vl, token_size, t_pad))

      return data_field

    def get_static_field( self, field, token_size = [-1, -1]):

      #support for static fields from other data types
      data_type = self.data_type
      f = self.path + '/' + data_type + '/static/' +self.data_type + '_' + field + self.file_ext

      x = self.file_loader(f, self.grib_index[field], static=True)

      if self.smoothing > 0 :
        sm = self.smoothing
        blur = partial( cv.blur, ksize=(sm,sm), borderType=cv.BORDER_REFLECT_101)
        # x = torch.from_numpy( cv.blur( x.numpy(), (self.smoothing,self.smoothing)))
        x = torch.from_numpy( blur( x.numpy() ))

      x = tokenize( x, token_size)

      return x
