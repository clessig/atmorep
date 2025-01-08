from atmorep.applications.downscaling.evaluator import Evaluator
import time


if __name__ == "__main__":

    model_id = 'ep6zj2o4'
    model_epoch = 7

    downscaling_params  = { 
                                      'dates' : [
                                         [2021, 1, 10, 12] ,  
                                         [2021, 4, 11, 0], 
                                         [2021, 7, 11, 12],
                                         [2021, 10, 11, 12]
                                        ],
                                      'token_overlap': [0,0],
                                      'downscaling_time_stamps' : 'whole',  #one of 'whole','center','last'
                                      'with_pytest' : False }

    now = time.time()
    Evaluator.evaluate( model_id, downscaling_params, model_epoch)
    print("time", time.time() - now)
