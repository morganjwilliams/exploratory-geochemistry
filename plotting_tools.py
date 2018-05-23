import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd


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