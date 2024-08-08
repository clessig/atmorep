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

from atmorep.core.evaluator import Evaluator
import time

if __name__ == '__main__':

  # models for individual fields
  #model_id = '4nvwbetz'     # vorticity
  #model_id = 'oxpycr7w'     # divergence
  #model_id = '1565pb1f'     # specific_humidity
  #model_id = '3kdutwqb'     # total precip
  model_id = 'dys79lgw'     # velocity_u
  #model_id = '22j6gysw'     # velocity_v
  # model_id = '15oisw8d'     # velocity_z
  #model_id = '3qou60es'     # temperature (also 2147fkco)
  #model_id = '2147fkco'     # temperature (also 2147fkco)
 
  # multi-field configurations with either velocity or voritcity+divergence
  #model_id = '1jh2qvrx'     # multiformer, velocity
  # model_id = 'wqqy94oa'     # multiformer, vorticity
  #model_id = '3cizyl1q'     # 3 field config: u,v,T
  # model_id = '1v4qk0qx'     # pre-trained, 3h forecasting
  # model_id = '1m79039j'     # pre-trained, 6h forecasting
  #model_id='34niv2nu'
  # supported modes: test, forecast, fixed_location, temporal_interpolation, global_forecast,
  #                  global_forecast_range
  # options can be used to over-write parameters in config; some modes also have specific options, 
  # e.g. global_forecast where a start date can be specified
  
  #Add 'attention' : True to options to store the attention maps. NB. supported only for single field runs. 
  
  # BERT masked token model
  mode, options = 'BERT', {'years_test' : [2021], 'num_samples_validate' : 128, 'with_pytest' : True }

  # BERT forecast mode
  #mode, options = 'forecast', {'forecast_num_tokens' : 2, 'num_samples_validate' : 128, 'with_pytest' : True }

  #temporal interpolation 
  #idx_time_mask: list of relative time positions of the masked tokens within the cube wrt num_tokens[0]  
  #mode, options = 'temporal_interpolation', {'idx_time_mask': [5,6,7], 'num_samples_validate' : 128, 'with_pytest' : True}

  # BERT forecast with patching to obtain global forecast
#   mode, options = 'global_forecast', { 
#                                       'dates' : [[2021, 1, 10, 18]],
# #                                     #   'dates' : [ #[2021, 1, 10, 18]
# #                                     #      [2021, 1, 10, 12] , [2021, 1, 11, 0], [2021, 1, 11, 12], [2021, 1, 12, 0], [2021, 1, 12, 12], [2021, 1, 13, 0], 
# #                                     #      [2021, 4, 10, 12], [2021, 4, 11, 0], [2021, 4, 11, 12], [2021, 4, 12, 0], [2021, 4, 12, 12], [2021, 4, 13, 0], 
# #                                     #      [2021, 7, 10, 12], [2021, 7, 11, 0], [2021, 7, 11, 12], [2021, 7, 12, 0], [2021, 7, 12, 12], [2021, 7, 13, 0], 
# #                                     #      [2021, 10, 10, 12], [2021, 10, 11, 0], [2021, 10, 11, 12], [2021, 10, 12, 0], [2021, 10, 12, 12], [2021, 10, 13, 0]
# #                                     #  ], 
#                                       'token_overlap' : [0, 0],
#                                       'forecast_num_tokens' : 2,
#                                       'with_pytest' : True }

  file_path = '/gpfs/scratch/ehpc03/era5_y2010_2021_res025_chunk8.zarr'
  
  now = time.time()
  Evaluator.evaluate( mode, model_id, file_path, options)
  print("time", time.time() - now)