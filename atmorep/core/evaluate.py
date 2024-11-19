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

import os
from pathlib import Path
from atmorep.core.evaluator import Evaluator
from atmorep.config.config import UserConfig
import time

if __name__ == '__main__':

  # arXiv 2023: models for individual fields
  #model_id = '4nvwbetz'     # vorticity
  #model_id = 'oxpycr7w'     # divergence
  #model_id = '1565pb1f'     # specific_humidity
  #model_id = '3kdutwqb'     # total precip
  #model_id = 'dys79lgw'     # velocity_u
  #model_id = '22j6gysw'     # velocity_v
  #model_id = '15oisw8d'     # velocity_z
  #model_id = '3qou60es'     # temperature 
  #model_id = '2147fkco'     # temperature (also 2147fkco)
  
  # new runs 2024
  #model_id='j8dwr5qj' #velocity_u
  #model_id='0tlnm5up' #velocity_v
  #model_id='v63l01zu' #specific humidity 
  #model_id='9l1errbo' #velocity_z
  model_id='7ojls62c' #temperature 1024 
  
  # supported modes: test, forecast, fixed_location, temporal_interpolation, global_forecast,
  #                  global_forecast_range
  # options can be used to over-write parameters in config; some modes also have specific options, 
  # e.g. global_forecast where a start date can be specified

  #Add 'attention' : True to options to store the attention maps. NB. supported only for single field runs. 
  
  # BERT masked token model
  #mode, options = 'BERT', {'years_val' : [2021], 'num_samples_validate' : 128, 'with_pytest' : True}

  # BERT forecast mode
  mode, options = 'forecast', {'forecast_num_tokens' : 2, 'num_samples_validate' : 96, 'with_pytest' : False }

  #temporal interpolation 
  #idx_time_mask: list of relative time positions of the masked tokens within the cube wrt num_tokens[0]  
  #mode, options = 'temporal_interpolation', {'idx_time_mask': [5,6,7], 'num_samples_validate' : 128, 'with_pytest' : True}
  
  atmorep_project_dir = Path(os.environ["SLURM_SUBMIT_DIR"])
  print("Atmorep project dir:", atmorep_project_dir)
  user_config = UserConfig.from_path(atmorep_project_dir)

  # BERT forecast with patching to obtain global forecast
  # mode, options = 'global_forecast', { 
                                      # 'dates' : [[2021, 2, 10, 12]],
                                      # 'dates' : [
                                      #    [2021, 1, 10, 12] , [2021, 1, 11, 0], [2021, 1, 11, 12], [2021, 1, 12, 0], #[2021, 1, 12, 12], [2021, 1, 13, 0], 
                                      #    [2021, 4, 10, 12], [2021, 4, 11, 0], [2021, 4, 11, 12], [2021, 4, 12, 0], #[2021, 4, 12, 12], [2021, 4, 13, 0], 
                                      #    [2021, 7, 10, 12], [2021, 7, 11, 0], [2021, 7, 11, 12], [2021, 7, 12, 0], #[2021, 7, 12, 12], [2021, 7, 13, 0], 
                                      #    [2021, 10, 10, 12], [2021, 10, 11, 0], [2021, 10, 11, 12], #[2021, 10, 12, 0], [2021, 10, 12, 12], [2021, 10, 13, 0]
                                      #  ], 
                                      # 'token_overlap' : [0, 0],
                                      # 'forecast_num_tokens' : 2,
                                      # 'with_pytest' : False }
  
  now = time.time()
  Evaluator.evaluate( mode, model_id, options, user_config=user_config)
  print("time", time.time() - now)