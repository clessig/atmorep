# SPDX-FileCopyrightText: 2024 AtmoRep collaboration: European Centre for Medium-Range Forecasting (ECMWF), Jülich Supercomputing Center (JSC), European Center for Nuclear Research (CERN)
#
# SPDX-License-Identifier: MIT

"""
Collection of plotting methods. 
Date: November 2024
"""

# import packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

def get_cmap_norm(levels, cb_name: str = "PuOr_r", cb_range= (0., 1.)):
    """
    Get the colormap and norm-object for given levels and a given colorbar-name
    :param levels: level boundaries
    :param cb_name: name of colorbar 
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)
    nbounds = len(bounds)
    
    col_obj = plt.get_cmap(cb_name)
    col_obj = col_obj(np.linspace(*cb_range, nbounds)) 

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm

def plot_global_data(data: xr.DataArray, plt_fname: Path, lshow: bool = True, **plt_kwargs):
    """
    Plot global map of data.
    :param data: 2D xr.DataArray with latitude and longitude dimension
    :param plt_fname: filename where plot will be saved
    :param lshow: show plot (in Jupyter Notebook)
    :param **plt_kwargs: optional keyword arguments
                         - figsize (tuple): tuple of figure size (default: (12, 6)
                         - projection: cartopy projection-object used to create the map (default: ccrs.PlateCarree())
                         - transform: cartopy projection-object to transform the data (default: inherit from projection)
                         - cmap_name (str): name of matplotlib colormap (default: "coolwarm")
                         - levels (list, array): levels for colorbar (default: np.arange(-30, 31, 2))
                         - cmap_range (tuple): range of colormap to be used (default: (0., 1.)
                         - data_label (str): label to annotate plotted data (default: "Temperature [°C]")

    """                 
    figsize = plt_kwargs.pop("figsize", (12, 6))
    proj = plt_kwargs.pop("projection", ccrs.PlateCarree())
    transform = plt_kwargs.pop("transform", proj)
    cmap_name = plt_kwargs.pop("cmap_name", "coolwarm")
    levels = plt_kwargs.pop("levels", np.arange(-30, 31, 2))
    cb_range = plt_kwargs.pop("cmap_range", (0., 1.))
    data_label = plt_kwargs.pop("data_label", "Temperature [°C]")

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})

    # get custom colormap and norm
    cmap, norm = get_cmap_norm(levels, cmap_name, cb_range)
    
    data.plot(ax=ax, transform=transform, cmap=cmap, norm=norm, cbar_kwargs={'label': data_label}, **plt_kwargs)

    # add coast lines
    ax.coastlines(linewidth=0.5, edgecolor='k', alpha=0.5)
    
    # Set global extent
    ax.set_global()

    print(f"Save plot to '{plt_fname}'...")
    plt.savefig(plt_fname)
    
    # Show the plot
    if lshow:
        plt.show()


def plot_metric_line(data: xr.DataArray, metric: dict,
                     plt_fname: str, varname: str = "T2m", model_name= "AtmoRep", x_coord: str = "hour",
                     lshow: bool = True, **kwargs):
    """
    Create line plots of 2D-metric data (e.g. metric plotted against time)
    :param data: DataArray containing the metric values
    :param model_name: Name of model
    :param metric: Dictionary containing metric name and unit
    :param plt_fname: File name of plot
    :param varname: Name of variable that was evaluated
    :param x_coord: Name of coordinate along which metric is plotted
    :param kwargs: Keyword arguments for plotting
                   Valid keys are:
                    - figsize (tuple): tuple of figure size (default: (12, 6)
                    - "linestyle": linestyle of plot, default: "k-"
                    - "error_color": color of error bounds, default: "blue"
                    - "value_range": range of y-axis, default: (0., 4.)
                    - "fs": font size of labels, default: 16
                    - "ref_line": reference line to be plotted, default: None
                    - "ref_linestyle": linestyle of reference line, default: "k--"
                    - other valid arguments of ax.plot
    """
    # get some plot parameters
    figsize = kwargs.pop("figsize", (12, 6))
    linestyle = kwargs.pop("linestyle", "k-")
    err_col = kwargs.pop("error_color", "blue")
    val_range = kwargs.pop("value_range", (0., 4.))
    fs = kwargs.pop("fs", 16)
    ref_line = kwargs.pop("ref_line", None)
    ref_linestyle = kwargs.pop("ref_linestyle", "k--")
    xlabel = kwargs.pop("xlabel", "daytime [UTC]")

    fig, (ax) = plt.subplots(1, 1, figsize=figsize)

    # create line plot
    ax.plot(data[x_coord].values, data.values, linestyle, label=model_name, **kwargs)

    # add reference line if desired
    if ref_line is not None:
        nval = np.shape(data[x_coord].values)[0]
        ax.plot(data[x_coord].values, np.full(nval, ref_line), ref_linestyle)
    ax.set_ylim(*val_range)

    # label axis
    ax.set_xlabel(xlabel, fontsize=fs)
    metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
    ax.set_ylabel(f"{metric_name} {varname} [{metric_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not str(plt_fname).endswith(".png") else plt_fname
    
    print(f"Save plot to '{plt_fname}'...")
    plt.savefig(plt_fname)
    plt.tight_layout()

    # Show the plot
    if lshow:
        plt.show()
