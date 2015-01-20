"""
models/plotting.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Various methods for plotting attributes of a StochasticModel, including
spectra.
"""

import itertools

import numpy as np

import matplotlib.pyplot as pp
import matplotlib.colors as mplcolors
import matplotlib.gridspec as gridspec


def symlog(x):
    """
    Signed log-scaling of input by its distance from zero. log(abs(x)) * sign(x)

    :param x (float): input value.
    :return: symmetrically log-scaled value.
    """

    dtype = type(x) if type(x) != np.ndarray else x.dtype

    return np.log(abs(x) + np.finfo(dtype).eps) * np.sign(x)


def plot_twopane_axes(n=1, ylabels=None, yticks=None, ylimits=None):
    """
    Set up a GridSpec for an NxN lower left triangular grid of subplots, each
    of which has two vertically stacked panes.

    :param n: Number of rows/columns in the grid.
    :param ylabels (nested list): NxN list of dictionaries, where each dict has
        a 'top' and 'bottom' key, the values of which are the ylabels for the
        corresponding panes (default: None).
    :param yticks (dict): dictionary with 'top' and 'bottom' keys indicating y
        tick values for top and bottom panes, assuming they are the same across
        all grid cells (default: None).
    :param ylimits (dict): dictionary with 'top' and 'bottom' keys pointing to
        two-value tuples indicating lower and upper y axis limits, assuming
        they are the same across all grid cells (default: None).
    :return (list): list of axes of instance matplotlib.Axes.
    """

    fig = pp.gcf()
    gs = gridspec.GridSpec(n, n, wspace=0.25, hspace=0.25)
    axlist = [[{} for j in range(n)] for i in range(n)]

    for i, j in itertools.combinations_with_replacement(range(n), 2):
        # This is the vertically stacked grid inside the current grid cell.
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[n * j + i], wspace=0, hspace=0
        )

        for pane_index, pane in enumerate(['top', 'bottom']):
            ax = pp.Subplot(pp.gcf(), inner_grid[pane_index])

            # Both panes should share the lower pane's x axis.
            if pane is 'top':
                ax.get_xaxis().set_visible(False)

            fig.add_subplot(ax)
            pp.sca(ax)

            # Set y ticks, limits, and labels if they're specified.
            if yticks[pane] is not None:
                pp.yticks(yticks[pane].values())
                ax.set_yticklabels(yticks[pane].keys())

            if ylimits is not None and pane in ylimits:
                pp.ylim(*ylimits[pane])

            if ylabels is not None:
                pp.ylabel(ylabels[i][j][pane])

            axlist[i][j][pane] = ax

    return axlist


def plot_cospectra(
        freqs,
        pxx,
        varnames=None,
        plotfun=None,
        axpanes=None,
        **plotargs
):
    """
    Make a NxN lower triangular plot of cospectra between state variables.

    :param freqs (array): (F,) List of frequencies.
    :param pxx (array): The (N,N,F) spectral matrix.
    :param varnames (list): List of N names for the states. If None, no labels
        will be drawn (default: None).
    :param plotfun (callable or str): Plot function to use for the spectrum. If
        None, pyplot.scatter will be used for frequency resolution F <= 8192,
        and pp.hist2d will be used for F > 8192 (default: None).
    :param axpanes (list): nested NxN list of dictionaries where each dict has
        keys 'top' and 'bottom' that point to matplotlib.Axes instances
        corresponding to the top (magnitude) and bottom (phase) plots
        respectively. If None is specified, the axes will be created and
        returned for re-use in overlaying plots.
    :param plotargs (dict): additional arguments to pass to plotfun.
    :return (list): nested NxN list of dictionaries formed in the same way as
        axpanes.
    """

    # Set up defaults for plotting type/parameters.
    if plotfun is None:
        # If auto, automatically determine if a plot should use scatter or
        # hist2d based on how many bins there are to plot.
        if len(freqs) > 8192:
            plotfun = pp.hist2d
        else:
            plotfun = pp.scatter

    if plotfun is pp.hist2d:
        plotargs.setdefault('bins', 200)
        plotargs.setdefault('normed', True)

        # If "color" is specified, make a colormap that linearly scales alpha
        # at that color. Otherwise, use the default or specified cmap. Useful
        # for overlaying multiple plots with different colors.
        if 'color' in plotargs:
            if 'cmap' not in plotargs:
                plotargs['cmap'] = mplcolors.LinearSegmentedColormap.from_list(
                    'custom_histcmap',
                    [
                        mplcolors.colorConverter.to_rgba(plotargs['color'], 0),
                        mplcolors.colorConverter.to_rgba(plotargs['color'], 1)
                    ]
                )

            plotargs.pop('color')
    elif plotfun is pp.scatter:
        plotargs.setdefault('alpha', 0.25)
        plotargs.setdefault('marker', '.')
        plotargs.setdefault('s', 1)

    # Make axes for two-pane lower left triangle grid if they aren't provided.
    n = len(pxx)

    if axpanes is None:
        axpanes = plot_twopane_axes(
            n=n,
            ylabels=[
                [
                    dict(
                        top='$\\log f_{%s%s}$' % (varnames[i], varnames[j]),
                        bottom='$\\angle f_{%s%s}$' % (varnames[i], varnames[j])
                    )
                    for j in range(n)
                ]
                for i in range(n)
            ] if varnames is not None else None,
            yticks=dict(
                top=None,
                bottom=dict(zip(
                    ['$-pi/2$', '$0$', '$pi/2$'],
                    [-np.pi/2, 0, np.pi/2]
                ))
            ),
            ylimits=dict(bottom=[-np.pi, np.pi])
        )

    # Plot each spectral matrix component in the created/provided axes.
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        # Magnitude.
        pp.sca(axpanes[i][j]['top'])
        plotfun(freqs, abs(pxx[i, j]), **plotargs)

        if np.sum(abs(pxx[i, j])) > 0:
            pp.yscale('log')

        pp.xlim(0, freqs[-1])

        # Phase.
        pp.sca(axpanes[i][j]['bottom'])
        plotfun(freqs, np.angle(pxx[i, j]), **plotargs)
        pp.xlim(0, freqs[-1])

    return axpanes


def plot_phase(series, varnames=None, logscale=False, plotfun=None, **plotargs):
    """
    Make a NxN lower triangular phase-space plot that plots pairs of state
    variables as functions of each other in a two-dimensional histogram.

    :param series (array): The (N,T) state trajectory.
    :param varnames (list): N state variable names. If None, no labels will be
        drawn (default: None).
    :param logscale (bool): Whether or not variables should be plot on a log
        scale (default: False).
    :param plotfun (callable): Matplotlib plotting function to use for the
        plots. If None, a plotting method will automatically be chosen from
        pp.plot, pp.scatter, and pp.hist2d, based upon how many data points
        there are (default: None).
    :param plotargs: Any additional parameters to plotfun, such as bin count.
    """

    # Set up defaults for plotting type/parameters.
    if plotfun is None:
        if len(series.shape) > 1 and series.shape[1] > 8192:
            plotfun = pp.hist2d
        else:
            plotfun = pp.scatter

    if plotfun is pp.scatter:
        plotargs.setdefault('alpha', 0.25)
        plotargs.setdefault('marker', '.')
        plotargs.setdefault('s', 1)
    elif plotfun is pp.hist2d:
        plotargs.setdefault('bins', 200)
        plotargs.setdefault('normed', True)

        if 'cmap' not in plotargs:
            plotargs.setdefault('color', 'green')

            plotargs['cmap'] = mplcolors.LinearSegmentedColormap.from_list(
                'custom_histcmap',
                [
                    mplcolors.colorConverter.to_rgba(plotargs['color'], 0),
                    mplcolors.colorConverter.to_rgba(plotargs['color'], 1)
                ]
            )

            plotargs.pop('color')

    nstates, nsamples = series.shape

    # Plot lower left triangle grid.
    for i, j in itertools.combinations_with_replacement(range(nstates), 2):
        pp.subplot(nstates, nstates, nstates * j + i + 1)

        # Plot axis labels if varname are provided.
        if varnames is not None:
            if j is len(varnames) - 1:
                pp.xlabel('$%s$' % varnames[i])

            if i is 0:
                pp.ylabel('$%s$' % varnames[j])

        # Scale values.g
        xs, ys = series[i], series[j]
        if logscale:
            xs, ys = symlog(xs), symlog(ys)

        plotfun(xs, ys, **plotargs)
