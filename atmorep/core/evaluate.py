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

if __name__ == '__main__':

  # models for individual fields
  model_id = '4nvwbetz'     # vorticity
  # model_id = 'oxpycr7w'     # divergence
  # model_id = '1565pb1f'     # specific_humidity
  # model_id = '3kdutwqb'     # total precip
  # model_id = 'dys79lgw'     # velocity_u
  # model_id = '22j6gysw'     # velocity_v
  # model_id = '15oisw8d'     # velocity_z
  # model_id = '3qou60es'     # temperature (also 2147fkco)
  # model_id = '2147fkco'     # temperature (also 2147fkco)

  # multi-field configurations with either velocity or voritcity+divergence
  # model_id = '1jh2qvrx'     # multiformer, velocity
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
  
  # BERT forecast mode
  mode, options = 'forecast', {'forecast_num_tokens' : 1} #, 'fields[0][2]' : [123, 137], 'attention' : False }
  
  # BERT forecast with patching to obtain global forecast
  # mode, options = 'global_forecast', { 'fields[0][2]' : [123, 137],
  #                                      'dates' : [[2021, 2, 10, 12]],
  #                                      'token_overlap' : [0, 0],
  #                                      'forecast_num_tokens' : 1,
  #                                      'attention' : False }

  Evaluator.evaluate( mode, model_id, options)
