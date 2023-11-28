# AtmoRep

This repository contains the source code for the [AtmoRep](https://www.atmorep.org) models for large scale representation learning of atmospheric dynamics as well as links to the pre-trained models and the required model input data.

The pre-print for the work is available on ArXiv: https://arxiv.org/abs/2308.13280.

```
@misc{Lessig2023atmorep,
  title = {AtmoRep: A stochastic model of atmosphere dynamics using large scale representation learning},
	author = {Christian Lessig and Ilaria Luise and Bing Gong and Michael Langguth and Scarlet Stadler and Martin Schultz},
  eprint = {2308.13280},
  primaryclass = {physics.ao-ph},
  url = {https://arxiv.org/abs/2308.13280},
  year = {2023},
```

# Starter README

## 1. Pull code

`````
%> wget git@github.com:clessig/atmorep.git
`````
This creates a directory ``atmorep`` with the code that contains the source code including the python scripts for model training and evaluation.

After following the steps described below, the final directory structure will look as follows:
````
└── atmorep/
    ├── atmorep/
    │     └── ... 
    ├── data/                         <- top level data directory
    │    ├── normalisation/           <- directory for data normalisations 
    │    ├── vorticity/
    │    │       ├── ml105/           <- model levels with monthly GRIB files
    │    │       │     ├── era5_vorticity_y2021_m03_ml137.grib   <- grib data file
    │    │       │     ├── ...
    │    │       ├── ml114/
    │    │       ├── ml123/
    │    │       ├── ml137/
    │    │       ├── ml96/
    .    .       .
    │    ├── temperature/
    .    .
    ├── models
    │    ├── id4nvwbetz   <- Directory containing model weights and config
    │    │       ├──  model_id4nvwbetz.json     
    │    │       └──  AtmoRep_id4nvwbetz.mod
    │    ├── id<model_id>
    .    .
    └── results
         ├── id4nvwbetz
         ...
````
The directories ``data``, ``models``, and ``results`` need to be created if they do not exist. All directories might be large and should thus be on a directory with sufficient storage space; in this case they can be soft-linked to the default ones above or they can be set in ``atmorep/config/config``. 

## 4. Download the data 

### 4.1 Download pre-trained models

Models can be downloaded from: https://datapub.fz-juelich.de/atmorep/trained-models.html

An example for downloading the pre-trained models is given here, in this case for the vorticity model.

`````
% atmorep/> mkdir models
% atmorep/> cd models
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/models/model_id4nvwbetz.tar.gz
% atmorep/data/> tar xvzf model_id4nvwbetz.tar.gz
% atmorep/data/> ls id4nvwbetz
AtmoRep_id4nvwbetz.mod  model_id4nvwbetz.json
`````


### 4.2 Download model input data (ERA5)

The input data in the required structure can be downloaded from the [Jülich datapub](https://datapub.fz-juelich.de/atmorep/era5-data.html) server. Direct link to WebDAV [https://datapub.fz-juelich.de/atmorep/data/](https://datapub.fz-juelich.de/atmorep/data/). Alternatively, it can be directly downloaded from MARS using the following [script](https://www.atmorep.org/code/mars_era5_download.py).

#### Download a subset of files

All data files (fields and normalizations) should be downloaded into the ``data`` directory. Un-taring the files will generate the correct folder structure. For example (we will use the vorticity example also below to run the first model so it is recommended to download it as a first step):
`````
% atmorep/> mkdir data
% atmorep/> cd data
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/data/vorticity/ml137/era5_vorticity_y2021_ml137.tar
% atmorep/data/> tar xvf era5_vorticity_y2021_ml137.tar
% atmorep/data/> ls -lah vorticity/ml137/
total 18G
era5_vorticity_y2021_m01_ml137.grib
era5_vorticity_y2021_m02_ml137.grib
...
era5_vorticity_y2021_m12_ml137.grib
`````
For efficiency reasons, AtmoRep takes monthly ERA5 data as input. Therefore, each tar file contains 12 GRIB files of about 1.5 GBytes each.

Coefficients for data normalization per field and level can be downloaded here: https://datapub.fz-juelich.de/atmorep/data/normalization/. They should also be located in the ```data``` directory:
`````
% atmorep/data/> wget https://datapub.fz-juelich.de/atmorep/data/normalization/normalization_vorticity_ml137.tar.gz
% atmorep/data/> tar xvzf normalization_vorticity_ml137.tar.gz
`````

## 5. Install python packages

Create a python environment, e.g.

`````
% atmorep/> python3 -m venv pyenv
`````

and activate the environment:

`````
% atmorep/> source pyenv/bin/activate
`````
conda is also possible, no environment is strictly required although we would recommend it. Please make sure to use a recent python version (we tested with python3.10).
Then install the AtmoRep package:
`````
% atmorep/>
% atmorep/> pip install -e .
`````

torch is currently not included (since it is often available or has particular dependencies, e.g. a specific Cuda version). In the simplest case, it can just be installed by:

`````
% atmorep/> pip install torch
`````
We require torch 2.x. (A container solution allows to run even on systems where torch 2.x is not available.)

## 6. Run model:
Pre-trained models can normally be run by:
`````
% atmorep/> python atmorep/core/evaluate.py
`````
You can easily adapt the configuration by selecting the corresponding _model_id_ in ``evaluate.py`` (see below). It defaults to the single-field configuration of vorticity, of which we have downloaded the data above.

Depending on your compute hardware, you might also have to run the computations by submitting the job using a batch system or allocate a compute node in interactive mode (if an interactive seesion is possible, then this is recommended). If you run an interactive session you will likely need to use the following:
`````
%  atmorep/> export CUDA_VISIBLE_DEVICES=0,1,2,3
%  atmorep/> MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
`````

The default evaluation mode is currently global forecast. The output will be (similar to) this:
````
devices : ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
Wandb run: atmorep-ztvyw7k6-8932958
Running Evaluate.evaluate with mode = global_forecast
Loaded AtmoRep id=4nvwbetz, ignoring/missing 2 elements.
Loaded model id = 4nvwbetz at epoch = -2.
Number of batches per global forecast: 14
INFO:: data stats vorticity : 5.374998363549821e-05 / 0.9978392720222473
num_accs_per_task : 1
with_hvd : True
hvd_rank : 0

...

wandb_id : ztvyw7k6
dates : [[2021, 2, 10, 12]]
token_overlap : [0, 0]
forecast_num_tokens : 1
validation loss for strategy=forecast at epoch 0 : 0.12402566522359848
validation loss for vorticity : 0.12402566522359848
wandb: Waiting for W&B process to finish... (success).
wandb: 
wandb: Run history:
wandb:        val. loss forecast ▁
wandb: val., forecast, vorticity ▁
wandb: 
wandb: Run summary:
wandb:        val. loss forecast 0.12403
wandb: val., forecast, vorticity 0.12403
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /p/project/atmo-rep/lessig/atmorep/atmorep/lessig-cleanup/atmorep/wandb/offline-run-20231124_095428-ztvyw7k6
````
For the vorticity example above, we evaluate with ``global_forecast`` for a specific date and using only a single model level:
````
mode, options = 'global_forecast', { 'fields[0][2]' : [137],
                                     'dates' : [ [2021, 2, 10, 12] ],
                                     'token_overlap' : [0, 0],
                                     'forecast_num_tokens' : 1, 
                                     'attention' : False}
````
We perform a 3 hour forecast, since 1 token is 3 hours wide. Another mode is the BERT masked token model mode used for pre-training:
`````
mode, options = 'BERT', {'years_test' : [2021], 'fields[0][2]' : [123, 137]}
`````
Again, we chose some custom options by using two levels instead of the five ones that are default and were used during pre-training and by using 2021 as the test year (since we downloaded the data).

The generated model output (stored in ``./results/id{wandbid}``) for the ```global_forecast``` example can be post-processed into a spatial map with the [following code](https://www.atmorep.org/code/plot_forecast.py). The run_id at the top needs to be replaced by the wandb_id of your run, it can be read off from the console output. Results will be stored as ``example_0000{0,1,2}.png``. The code is also an as-simple-as-possible example with many parameters hard-coded, see our analysis code for a proper handling. 
