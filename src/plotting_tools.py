import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np


def colorbar(mappable, **kwargs):
    """
    http://joseph-long.com/writing/colorbars/
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)


def plot_latlong(df,
                 bmp,
                 ax=None,
                 latvar='Latitude',
                 longvar='Longitude',
                 **kwargs):
    """
    Plots sample locations on a basemap axes.
    """
    lats, lons = df.loc[:, latvar].values, df.loc[:, longvar].values
    x, y = bmp(lons, lats)
    if not ax is None:
        ax.scatter(x, y,**kwargs)
    else:
        plt.gca().scatter(x, y, **kwargs)

        
def age_distribution(df,
                     yvar=None,
                     ax=None,
                     agerange=(0, 4.567 * 10**3),
                     bins=150,
                     agevar='Age',
                     **kwargs):
    """
    Plots the age distribution of samples. Plot is either a univariate histogram
    or a bivariate plot with age as the independent axis.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    else:
        fig = ax.figure
    ax.set_xlabel('Age (Ma)')
    agebins = np.linspace(*agerange, num=bins)

    if yvar is None:
        ret = ax.hist(df.loc[:, agevar], bins=agebins, range=agerange, **kwargs)
    else:
        ax.set_ylabel(yvar)
        ret = ax.scatter(df.loc[:, agevar], df.loc[:, yvar], **kwargs)
    return fig, ax, ret


def cleanup_ternary(tax):
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple=20, color="k")
    tax.ticks(axis='lbr', linewidth=1, multiple=20, clockwise=False, fontsize=8)
    a = tax.get_axes()
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)
    a.get_xaxis().set_ticks([])
    a.get_yaxis().set_ticks([])
    
    

def draw_vector(v0, v1,
                ax=None,
                **kwargs):
    """
    Plots an arrow represnting the direction and magnitue of a principal
    component on a biaxial plot.

    Todo: update for ternary plots.

    Modified after Jake VanderPlas' Python Data Science Handbook
    https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    """
    ax = ax
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    arrowprops.update(kwargs)
    ax.annotate('', v1, v0, arrowprops=arrowprops)


def vector_to_line(mu:np.array,
                   vector:np.array,
                   variance:float,
                   spans: int=4,
                   expand: int=10):
    """
    Creates an array of points representing a line along a vector - typically
    for principal component analysis.

    Modified after Jake VanderPlas' Python Data Science Handbook
    https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    """
    length = np.sqrt(variance)
    parts = np.linspace(-spans, spans, expand * 2 * spans + 1)
    line =  length * np.dot(parts[:, np.newaxis], vector[np.newaxis, :]) + mu
    line =  length * parts.reshape(parts.shape[0], 1) * vector + mu
    return line