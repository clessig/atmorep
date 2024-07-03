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

  #2 nodes
  #model_id='1b43bynq'
  # model_id='p20z3ilu'
  # model_id='10y42b1u'
  #model_id='99tb5lcy'
  # model_id='sn6h8wvq'
  # model_id='085saknn'

  #1node train continue
  #model_id='h7orvjna'
  #model_id='ocpn87si'

  # 1node
  #temperature
  #model_id='66zlffty'
  #model_id='fmzy4mxr'
  model_id='ezi4shmb'

  #velocity_u
  #model_id='hg8cy3c4'
  #model_id='av0rp1mj'
  #model_id='fc5o31h2'

  #specific_humidity
  #model_id='w965qy0o'
  #model_id='gpksrtrl'
  #model_id='c6am1m3j'

  #model_id = '4nvwbetz'     # vorticity
  #model_id = 'oxpycr7w'     # divergence
  #model_id = '1565pb1f'     # specific_humidity
  #model_id = '3kdutwqb'     # total precip
  #model_id = 'dys79lgw'     # velocity_u
  #model_id = '22j6gysw'     # velocity_v
  # model_id = '15oisw8d'     # velocity_z
  #model_id = '3qou60es'     # temperature (also 2147fkco)
  #model_id = '2147fkco'     # temperature (also 2147fkco)

  # multi-field configurations with either velocity or voritcity+divergence
  #model_id = '1jh2qvrx'     # multiformer, velocity
  # model_id = 'wqqy94oa'     # multiformer, vorticity
  # model_id = '3cizyl1q'     # 3 field config: u,v,T
  # model_id = '1v4qk0qx'     # pre-trained, 3h forecasting
  # model_id = '1m79039j'     # pre-trained, 6h forecasting
  
  # supported modes: test, forecast, fixed_location, temporal_interpolation, global_forecast,
  #                  global_forecast_range
  # options can be used to over-write parameters in config; some modes also have specific options, 
  # e.g. global_forecast where a start date can be specified
  
  # BERT masked token model
  #mode, options = 'BERT', {'years_test' : [2021], 'fields[0][2]' : [123, 137], 'attention' : False}
  #mode, options = 'BERT', {'years_test' : [2021], 'fields[0][2]' : [123], 'attention' : False}
  mode, options = 'BERT', {'years_test' : [2021], 'attention' : False}
  # BERT forecast mode
  #mode, options = 'forecast', {'forecast_num_tokens' : 2, 'fields[0][2]' : [123], 'attention' : False }
  #mode, options = 'forecast', {'forecast_num_tokens' : 2, 'attention' : False }
  #temporal interpolation 
  #idx_time_mask: list of relative time positions of the masked tokens within the cube wrt num_tokens[0]  
  #mode, options = 'temporal_interpolation', {'fields[0][2]' : [123], 'idx_time_mask': [5,6,7],  'attention' : False }

  # BERT forecast with patching to obtain global forecast
  # mode, options = 'global_forecast', { #'fields[0][2]' : [114], #[123, 137], #[105, 137],
  #                                     #'dates' : [[2021, 1, 10, 18]],
  #                                     'dates' : [ [2021, 1, 10, 18]
  #                                       # [2021, 1, 10, 12] , [2021, 1, 11, 0], [2021, 1, 11, 12], [2021, 1, 12, 0], [2021, 1, 12, 12], [2021, 1, 13, 0], 
  #                                       # [2021, 4, 10, 12], [2021, 4, 11, 0], [2021, 4, 11, 12], [2021, 4, 12, 0], [2021, 4, 12, 12], [2021, 4, 13, 0], 
  #                                       # [2021, 7, 10, 12], [2021, 7, 11, 0], [2021, 7, 11, 12], [2021, 7, 12, 0], [2021, 7, 12, 12], [2021, 7, 13, 0], 
  #                                       # [2021, 10, 10, 12], [2021, 10, 11, 0], [2021, 10, 11, 12], [2021, 10, 12, 0], [2021, 10, 12, 12], [2021, 10, 13, 0]
  #                                     ], 
  #                                     'token_overlap' : [0, 0],
  #                                     'forecast_num_tokens' : 2,
  #                                     'attention' : False }
  now = time.time()
  Evaluator.evaluate( mode, model_id, options)
  print("time", time.time() - now)
