# SPDX-FileCopyrightText: 2024 AtmoRep collaboration: European Centre for Medium-Range Forecasting (ECMWF), Jülich Supercomputing Center (JSC), European Center for Nuclear Research (CERN)
#
# SPDX-License-Identifier: MIT


"""
Collection of useful metrics to evaluate the performance of the predictions
Credits to: AtmoRep collaboration
Date: July 2023
"""

# import packages
import sys
sys.path.append("./")
try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from typing import Union, List

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
str_or_list = Union[str, List[str]]


##########################################

def calc_scores_item(pred, target, ens, scores, options, avg = []):
    score_engine = Scores(pred, target, ens, avg_dims = avg)
    score_list = [score_engine(score, **options) for score in scores]
    return score_list

##########################################

def get_cdf_of_x(sample_in, prob_in):
    '''
    Wrappper for interpolating CDF-value for given data
    :param sample_in : input values to derive discrete CDF
    :param prob_in   : corresponding CDF
    :return : lambda function converting arbitrary input values to corresponding CDF value
    '''
    return lambda xin: np.interp(xin, sample_in, prob_in)

##########################################

def get_seeps_matrix(seeps_param):
    """
    Converts SEEPS paramter array to SEEPS matrix.
    :param seeps_param: Array providing p1 and p3 parameters of SEEPS weighting matrix.
    """
    # initialize matrix
    seeps_weights = xr.full_like(seeps_param["p1"], np.nan)
    seeps_weights = seeps_weights.expand_dims(dim={"weights":np.arange(9)}, axis=0).copy()
    seeps_weights.name = "SEEPS weighting matrix"
    
    # off-diagonal elements
    seeps_weights[{"weights": 1}] = 1./(1. - seeps_param["p1"])
    seeps_weights[{"weights": 2}] = 1./seeps_param["p3"] + 1./(1. - seeps_param["p1"])
    seeps_weights[{"weights": 3}] = 1./seeps_param["p1"]
    seeps_weights[{"weights": 5}] = 1./seeps_param["p3"]
    seeps_weights[{"weights": 6}] = 1./seeps_param["p1"] + 1./(1. - seeps_param["p3"])
    seeps_weights[{"weights": 7}] = 1./(1. - seeps_param["p3"])
    # diagnol elements
    seeps_weights[{"weights": [0, 4, 8]}] = xr.where(np.isnan(seeps_weights[{"weights": 7}]), np.nan, 0.)
    
    return seeps_weights

##########################################

def perform_block_bootstrap_metric(metric: da_or_ds, dim_name: str, block_length: int, nboots_block: int = 1000,
                                   seed: int = 42):
    """
    Performs block bootstrapping on metric along given dimension (e.g. along time dimension).
    Note: Requires that the metric is linear over selected dimension.
    :param metric: DataArray or dataset of metric that should be bootstrapped
    :param dim_name: name of the dimension on which division into blocks is applied
    :param block_length: length of block (index-based)
    :param nboots_block: number of bootstrapping steps to be performed
    :param seed: seed for random block sampling (to be held constant for reproducability)
    :return: bootstrapped version of metric(-s)
    """

    method = perform_block_bootstrap_metric.__name__

    if not isinstance(metric, da_or_ds.__args__):
        raise ValueError("%{0}: Input metric must be a xarray DataArray or Dataset and not {1}".format(method,
                                                                                                       type(metric)))
    if dim_name not in metric.dims:
        raise ValueError("%{0}: Passed dimension cannot be found in passed metric.".format(method))

    metric = metric.sortby(dim_name)

    dim_length = np.shape(metric.coords[dim_name].values)[0]
    nblocks = int(np.floor(dim_length/block_length))

    if nblocks < 10:
        raise ValueError("%{0}: Less than 10 blocks are present with given block length {1:d}."
                         .format(method, block_length) + " Too less for bootstrapping.")

    # precompute metrics of block
    for iblock in np.arange(nblocks):
        ind_s, ind_e = iblock * block_length, (iblock + 1) * block_length
        metric_block_aux = metric.isel({dim_name: slice(ind_s, ind_e)}).mean(dim=dim_name)
        if iblock == 0:
            metric_val_block = metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0).copy(deep=True)
        else:
            metric_val_block = xr.concat([metric_val_block, metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0)],
                                         dim="iblock")

    metric_val_block["iblock"] = np.arange(nblocks)

    # get random blocks
    np.random.seed(seed)
    iblocks_boot = np.sort(np.random.randint(nblocks, size=(nboots_block, nblocks)))

    print("%{0}: Start block bootstrapping...".format(method))
    iterator_b = np.arange(nboots_block)
    if l_tqdm:
        iterator_b = tqdm(iterator_b)
    for iboot_b in iterator_b:
        metric_boot_aux = metric_val_block.isel(iblock=iblocks_boot[iboot_b, :]).mean(dim="iblock")
        if iboot_b == 0:
            metric_boot = metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0).copy(deep=True)
        else:
            metric_boot = xr.concat([metric_boot, metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0)], dim="iboot")

    # set iboot-coordinate
    metric_boot["iboot"] = np.arange(nboots_block)
    if isinstance(metric_boot, xr.Dataset):
        new_varnames = ["{0}_bootstrapped".format(var) for var in metric.data_vars]
        metric_boot = metric_boot.rename(dict(zip(metric.data_vars, new_varnames)))

    return metric_boot

##########################################

class Scores:
    """
    Class to calculate scores and skill scores.
    """
    def __init__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, data_ens: xr.DataArray,  avg_dims: str_or_list = "all"):
        """
        Constructor of score engine.
        :param data_fcst: forecast data to evaluate 
        :param data_ref: reference or ground truth data
        :param avg_dims: dimension or list of dimensions over which scores shall be averaged. 
                         Parse 'all' to average over all data dimensions.
        """
        self.metrics_dict = {"ets": self.calc_ets, "pss": self.calc_pss, "fbi": self.calc_fbi,
                             "mae": self.calc_mae, "l1": self.calc_l1, "l2": self.calc_l2, 
                             "mse": self.calc_mse, "rmse": self.calc_rmse, "bias": self.calc_bias,
                             "acc": self.calc_acc, "bias": self.calc_bias, "spread" : self.calc_spread, 
                             "ssr": self.calc_ssr, "grad_amplitude": self.calc_spatial_variability,
                             "psnr": self.calc_psnr, "iqd": self.calc_iqd, "seeps": self.calc_seeps} 
        self.data_fcst = data_fcst
        self.data_dims = list(self.data_fcst.dims)
        self.data_ref = data_ref
        self.data_ens = data_ens
        self.avg_dims = avg_dims

    def __call__(self, score_name, **kwargs):
        try:
            score_func = self.metrics_dict[score_name]
        except:
            raise ValueError(f"{score_name} is not an implemented score." +
                             "Choose one of the following: {0}".format(", ".join(self.metrics_dict.keys())))

        return score_func(**kwargs)

    @property
    def data_fcst(self):
        return self._data_fcst

    @data_fcst.setter
    def data_fcst(self, da_fcst):
        if not isinstance(da_fcst, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        self._data_fcst = da_fcst

    @property
    def data_ref(self):
        return self._data_ref

    @data_ref.setter
    def data_ref(self, da_ref):
        if not isinstance(da_ref, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        if not list(da_ref.dims) == self.data_dims:
            raise ValueError("Dimensions of data_fcst and data_ref must match, but got:" +
                             "[{0}] vs. [{1}]".format(", ".join(list(da_ref.dims)),
                                                      ", ".join(self.data_dims)))

        self._data_ref = da_ref

    @property
    def avg_dims(self):
        return self._avg_dims

    @avg_dims.setter
    def avg_dims(self, dims):
        if dims is None:
            self._avg_dims = None
        elif dims == "all":
            self._avg_dims = self.data_dims
            # print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if not x]
                raise ValueError("The following dimensions for score-averaging are not " +
                                 "part of the data: {0}".format(", ".join(np.array(dims)[ind_bad])))

            self._avg_dims = dims

    def get_2x2_event_counts(self, thresh):
        """
        Get counts of 2x2 contingency tables
        """
        a = ((self.data_fcst >= thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        b = ((self.data_fcst >= thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)
        c = ((self.data_fcst < thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        d = ((self.data_fcst < thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)

        return a, b, c, d

    def calc_ets(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)
        n = a + b + c + d
        ar = (a + b)*(a + c)/n      # random reference forecast
        
        denom = (a + b + c - ar)

        ets = (a - ar)/denom
        ets = ets.where(denom > 0, np.nan)

        return ets
    
    def calc_fbi(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)

        denom = a+c
        fbi = (a + b)/denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi
    
    def calc_pss(self, thresh=0.1):
        a, b, c, d = self.get_2x2_event_counts(thresh)      

        denom = (a + c)*(b + d)
        pss = (a*d - b*c)/denom

        pss = pss.where(denom > 0, np.nan)

        return pss   

    def calc_l1(self, **kwargs):
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        L1 will be divided by the number of samples along the average dimensions.
        Similar to MAE, but provides just a number divided by number of samples along average dimensions.
        :return: L1-error 
        """
        sum_dims = kwargs.get("sum_dims", [])   

        l1 = (np.abs(self.data_fcst - self.data_ref)).sum(dim=sum_dims)

        if self.avg_dims is not None:
            len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
            l1 /= np.prod(len_dims)

        return l1
    
    def calc_l2(self, **kwargs):
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.
        Similar to RMSE, but provides just a number divided by number of samples along average dimensions.
        :return: L2-error 
        """
        sum_dims = kwargs.get("sum_dims", [])   

        l2 = np.sqrt((np.square(self.data_fcst - self.data_ref)).sum(dim=sum_dims))

        
        if self.avg_dims is not None:
            len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
            l2 /= np.prod(len_dims)

        return l2

    def calc_mae(self, **kwargs):
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data
        :return: MAE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mae are without effect.")   

        if self.avg_dims is None:
            raise ValueError(f"Cannot calculate mean absolute error without average dimensions (avg_dims=None).")
        mae = np.abs(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mae

    def calc_mse(self, **kwargs):
        """
        Calculate mean squared error (MSE) of forecast data w.r.t. reference data
        :return: MSE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mse are without effect.")
        
        if self.avg_dims is None:
            raise ValueError(f"Cannot calculate mean squared error without average dimensions (avg_dims=None).")

        mse = np.square(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mse

    def calc_rmse(self, **kwargs):
        """
        Calculate root mean squared error (RMSE) of forecast data w.r.t. reference data
        :return: RMSE averaged over provided dimensions
        """
        if self.avg_dims is None:
            raise ValueError(f"Cannot calculate root mean squared error without average dimensions (avg_dims=None).")

        rmse = np.sqrt(self.calc_mse(**kwargs))

        return rmse
    
    def calc_acc(self, **kwargs):
        """
        Calculate anomaly correlation coefficient (ACC).
        :param clim_mean: climatological mean of the data
        :param spatial_dims: names of spatial dimensions over which ACC are calculated. 
                             Note: No averaging is possible over these dimensions.
        :return acc: Averaged ACC (except over spatial_dims)
        """
        clim_mean = kwargs.get("clim_mean")   
        spatial_dims = kwargs.get("spatial_dims", ["lat", "lon"])

        #fcst_ano, obs_ano = self.data_fcst - clim_mean, self.data_ref - clim_mean

        acc = ((self.data_fcst - clim_mean)*(self.data_ref - clim_mean)).sum(spatial_dims)/np.sqrt(((self.data_fcst - clim_mean)**2).sum(spatial_dims)*((self.data_ref - clim_mean)**2).sum(spatial_dims))

        if self.avg_dims is not None:
            mean_dims = [x for x in self.avg_dims if x not in spatial_dims]
            if len(mean_dims) > 0:
                acc = acc.mean(mean_dims)

        return acc

    def calc_spread(self, **kwargs):
        """
        Calculate the spread of the forecast ensemble 
        :return: spread averaged over the provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_spread are without effect.")

        ens_std = self.data_ens.std(dim = "ensemble")
        return np.sqrt((ens_std**2).mean(dim = self.avg_dims))

    def calc_ssr(self, **kwargs):
        """
        Calculate the Spread-Skill Ratio (SSR) of the forecast ensemble data w.r.t. reference data
        :return: the SSR averaged over provided dimensions
        """
        # ens_std = data_ens.std(dim = "ensemble")
        # spread = np.sqrt((ens_std**2).mean(dim = avg_dims))
       
        #spread = self.calc_spread(**kwargs)
        #mse = np.square(self.data_fcst - self.data_ref).mean(dim = self.avg_dims)
        #rmse = np.sqrt(mse)
        return self.calc_spread(**kwargs)/self.calc_rmse(**kwargs) #spread/rmse

    def calc_bias(self, **kwargs):
        """
        Calculate mean bias of forecast data w.r.t. reference data
        :return: bias averaged over provided dimensions
        """

        if kwargs:
            print("Passed keyword arguments to calc_bias are without effect.")

        bias = self.data_fcst - self.data_ref
        
        if self.avg_dims is not None:
            bias = bias.mean(dim=self.avg_dims)

        return bias

    def calc_psnr(self, **kwargs):
        """
        Calculate PSNR of forecast data w.r.t. reference data
        :param kwargs: known keyword argument 'pixel_max' for maximum value of data
        :return: averaged PSNR
        """
        pixel_max = kwargs.get("pixel_max", 1.)

        mse = self.calc_mse()
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.
        else:
            psnr = 20. * np.log10(pixel_max / np.sqrt(mse))

        return psnr

    def calc_spatial_variability(self, **kwargs):
        """
        Calculates the ratio between the spatial variability of differental operator with order 1 (or 2) forecast and
        reference data using the calc_geo_spatial-method.
        :param kwargs: 'order' to control the order of spatial differential operator
                       'non_spatial_avg_dims' to add averaging in addition to spatial averaging performed with calc_geo_spatial
        :return: the ratio between spatial variabilty in the forecast and reference data field
        """
        order = kwargs.get("order", 1)
        avg_dims = kwargs.get("non_spatial_avg_dims", None)

        fcst_grad = self.calc_geo_spatial_diff(self.data_fcst, order=order)
        ref_grd = self.calc_geo_spatial_diff(self.data_ref, order=order)

        ratio_spat_variability = (fcst_grad / ref_grd)
        if avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=avg_dims)

        return ratio_spat_variability

    def calc_iqd(self, align_join: str = "exact", steps: int = 1000) -> xr.DataArray:
        """
        Calculate the Integrated Quadratic Distance (IQD) between the forecast and reference data in CDF space.

        :param align_join (str, optional): How to align the datasets, cf. join-parameter of xarray.align-method
        :return: The Integrated Quadratic Distance.
        """
        # get local logger
        func_logger = logging.getLogger(f"{logger_module_name}.Scores.{self.calc_iqd.__name__}")

        # IQD evaluates the marginal distribution of the data and thus collapes existing dimensions
        # There, averaging is not meaningful here
        if self.avg_dims and self.avg_dims != []:
            func_logger.debug(f"Parsed averaging dimensions ({', '.join(self.avg_dims)}) are ignored.")

        # Align the arrays and sort the data incl. falttening
        forecast, reference = xr.align(self.data_fcst, self.data_ref, join=align_join)
        
        forecast, reference = np.sort(forecast, axis=None), np.sort(reference, axis=None)

        # get empirical CDF-functions for forecast and reference data
        npoints = len(forecast)
        cdf_val = 1. * np.arange(npoints)/ (npoints - 1)

        cdf_fcst = self.get_cdf_of_x(forecast, cdf_val)
        cdf_ref = self.get_cdf_of_x(reference, cdf_val)

        # get integration points        
        min_val, max_val = min(reference[0], forecast[0]), max(reference[-1], forecast[-1])
        
        xnodes = np.linspace(min_val, max_val, num=steps)
        
        # calculate CDF at integration ponts...
        cdf_fcst_x = cdf_fcst(xnodes)
        cdf_ref_x = cdf_ref(xnodes)  

        # ...and integrate squared difference for IQD
        iqd = np.trapz(np.square(cdf_ref_x - cdf_fcst_x), xnodes)

        return iqd

    def calc_seeps(self, seeps_weights: xr.DataArray, t1: xr.DataArray, t3: xr.DataArray, spatial_dims: List):
        """
        Calculates stable equitable error in probabiliyt space (SEEPS), see Rodwell et al., 2011
        :param seeps_weights: SEEPS-parameter matrix to weight contingency table elements
        :param t1: threshold for light precipitation events
        :param t3: threshold for strong precipitation events
        :param spatial_dims: list/name of spatial dimensions of the data
        :return seeps skill score (i.e. 1-SEEPS)
        """

        def seeps(data_ref, data_fcst, thr_light, thr_heavy, seeps_weights):
            ob_ind = (data_ref > thr_light).astype(int) + (data_ref >= thr_heavy).astype(int)
            fc_ind = (data_fcst > thr_light).astype(int) + (data_fcst >= thr_heavy).astype(int)
            indices = fc_ind * 3 + ob_ind  # index of each data point in their local 3x3 matrices
            seeps_val = seeps_weights[indices, np.arange(len(indices))]  # pick the right weight for each data point
            
            return 1.-seeps_val
        
        if self.data_fcst.ndim == 3:
            assert len(spatial_dims) == 2, f"Provide two spatial dimensions for three-dimensional data."
            data_fcst, data_ref = self.data_fcst.stack({"xy": spatial_dims}), self.data_ref.stack({"xy": spatial_dims})
            seeps_weights = seeps_weights.stack({"xy": spatial_dims})
            t3 = t3.stack({"xy": spatial_dims})
            lstack = True
        elif self.data_fcst.ndim == 2:
            data_fcst, data_ref = self.data_fcst, self.data_ref
            lstack = False
        else:
            raise ValueError(f"Data must be a two-or-three-dimensional array.")

        # check dimensioning of data
        assert data_fcst.ndim <= 2, f"Data must be one- or two-dimensional, but has {data_fcst.ndim} dimensions. Check if stacking with spatial_dims may help." 

        if data_fcst.ndim == 1:
            seeps_values_all = seeps(data_ref, data_fcst, t1.values, t3, seeps_weights)
        else:
            data_fcst, data_ref = data_fcst.transpose(..., "xy"), data_ref.transpose(..., "xy")
            seeps_values_all = xr.full_like(data_fcst, np.nan)
            seeps_values_all.name = "seeps"
            for it in range(data_ref.shape[0]):
                data_fcst_now, data_ref_now = data_fcst[it, ...], data_ref[it, ...]
                # in case of missing data, skip computation
                if np.all(np.isnan(data_fcst_now)) or np.all(np.isnan(data_ref_now)):
                    continue

                seeps_values_all[it,...] = seeps(data_ref_now, data_fcst_now, t1.values, t3, seeps_weights.values)

        if lstack:
            seeps_values_all = seeps_values_all.unstack()

        seeps_values = seeps_values_all.mean(dim=self.avg_dims)

        return seeps_values

    @staticmethod
    def calc_geo_spatial_diff(scalar_field: xr.DataArray, order: int = 1, r_e: float = 6371.e3, dom_avg: bool = True):
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2) of a scalar field given on a regular,
        geographical grid (i.e. dlambda = const. and dphi=const.)
        :param scalar_field: scalar field as data array with latitude and longitude as coordinates
        :param order: order of spatial differential operator
        :param r_e: radius of the sphere
        :return: the amplitude of the gradient/laplacian at each grid point or over the whole domain (see avg_dom)
        """
        method = Scores.calc_geo_spatial_diff.__name__
        # sanity checks
        assert isinstance(scalar_field, xr.DataArray), f"Scalar_field of {method} must be a xarray DataArray."
        assert order in [1, 2], f"Order for {method} must be either 1 or 2."

        dims = list(scalar_field.dims)
        lat_dims = ["rlat", "lat", "latitude"]
        lon_dims = ["rlon", "lon", "longitude"]

        def check_for_coords(coord_names_data, coord_names_expected):
            stat = False
            for i, coord in enumerate(coord_names_expected):
                if coord in coord_names_data:
                    stat = True
                    break

            if stat:
                return i, coord_names_expected[i]  # just take the first value
            else:
                raise ValueError("Could not find one of the following coordinates in the passed dictionary: {0}"
                                 .format(",".join(coord_names_expected)))

        lat_ind, lat_name = check_for_coords(dims, lat_dims)
        lon_ind, lon_name = check_for_coords(dims, lon_dims)

        lat, lon = np.deg2rad(scalar_field[lat_name]), np.deg2rad(scalar_field[lon_name])
        dphi, dlambda = lat[1].values - lat[0].values, lon[1].values - lon[0].values

        if order == 1:
            dvar_dlambda = 1. / (r_e * np.cos(lat) * dlambda) * scalar_field.differentiate(lon_name)
            dvar_dphi = 1. / (r_e * dphi) * scalar_field.differentiate(lat_name)
            dvar_dlambda = dvar_dlambda.transpose(*scalar_field.dims)  # ensure that dimension ordering is not changed

            var_diff_amplitude = np.sqrt(dvar_dlambda ** 2 + dvar_dphi ** 2)
            if dom_avg: var_diff_amplitude = var_diff_amplitude.mean(dim=[lat_name, lon_name])
        else:
            raise ValueError(f"Second-order differentation is not implemenetd in {method} yet.")

        return var_diff_amplitude

    @staticmethod
    def get_cdf_of_x(sample_in, prob_in):
        """
        Wrappper for interpolating CDF-value for given data
        :param sample_in : input values to derive discrete CDF
        :param prob_in   : corresponding CDF
        :return: lambda function converting arbitrary input values to corresponding CDF value
        """
        return lambda xin: np.interp(xin, sample_in, prob_in)
