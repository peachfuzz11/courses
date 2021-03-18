import numpy
from cartopy import crs
from matplotlib import pyplot
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from main.helpers.figurehelper import Figurehelper


def plot_global(lon, lat, data, point_size=10, title="", cbar_label="Br [mT]", cmap=pyplot.cm.PuOr_r):
    data = data * 10 ** (-6)
    lat = 90 - lat
    limit = numpy.max(abs(data))
    # create figure
    fig = pyplot.figure(figsize=(9, 5))
    # make array of axes
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[0.35, 0.65])
    axes = []
    axes.append(pyplot.subplot(gs[0, 0], projection=crs.NearsidePerspective(central_latitude=90.)))
    axes.append(pyplot.subplot(gs[0, 2], projection=crs.NearsidePerspective(central_latitude=-90.)))
    axes.append(pyplot.subplot(gs[1, :], projection=crs.Mollweide()))
    # Iterate over axes
    for ax in axes:
        pc = ax.scatter(lon, lat, c=data, s=point_size, cmap=cmap, vmin=-limit,
                        vmax=limit, transform=crs.PlateCarree())
        ax.gridlines(linewidth=0.5,
                     ylocs=numpy.linspace(-90, 90, num=7),  # parallels
                     xlocs=numpy.linspace(-180, 180, num=13),
                     color='grey', alpha=0.6, linestyle='-')  # meridians
        ax.coastlines(linewidth=0.5)
    # Add colorbar
    # inset axes into global map and move upwards
    cax = inset_axes(axes[-1], width="55%", height="10%", loc='upper center', borderpad=-9)
    # use last artist for the colorbar
    clb = pyplot.colorbar(pc, cax=cax, extend='both', orientation='horizontal')
    clb.set_label('{}'.format(cbar_label), fontsize=12)

    # Title
    pyplot.suptitle("{}".format(title))
    # Adjust plot
    pyplot.subplots_adjust(top=0.985, bottom=0.015, left=0.008, right=0.992, hspace=0.0, wspace=0.0)
    return Figurehelper(fig).to_png()


def plot_resolution(image, title=""):
    fig, ax = pyplot.subplots(figsize=(24, 12))
    crange = numpy.linspace(0, 1, 9)
    # colors = ['blue', 'yellow', 'green']
    # cmap = LinearSegmentedColormap.from_list("", colors)
    norm = BoundaryNorm(boundaries=crange, ncolors=256)
    im = ax.imshow(image, cmap='viridis', norm=norm,
                   interpolation=None)
    divider = make_axes_locatable(ax)
    ax.set_title(title + " trace = " + str(numpy.trace(image)))
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax)
    cbar.set_alpha(1)
    cbar.draw_all()
    # cbar.set_ticks([c + 0.1 for c in crange])
    return Figurehelper(fig).to_png()
