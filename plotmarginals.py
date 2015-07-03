"""
plotmarginals.py
parasynchrony
2015 Brandon Mechtley
Reuman Lab, Kansas Biological Survey
"""

import os
import sys
import cPickle
import itertools

import numpy as np
import scipy.stats

import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

import utilities


def plot_marginal(
        dim1, dim2, dimsum, hist,
        ax=None, axis=0, zdir='x', cmap='cubehelix'
):
    if ax is None:
        ax = pp.gca()

    mdim1, mdim2 = np.meshgrid(dim1, dim2)
    mdimsum = np.sum(hist, axis=axis).T / np.amax(hist)

    offset_idx = int(zdir is 'y') - 1

    ax.contourf(
        mdim1, mdim2, mdimsum,
        zdir=zdir,
        offset=dimsum[offset_idx] - np.diff(dimsum)[offset_idx] / 10,
        cmap=cmap,
        alpha=0.5
    )


def plot_marginals(xs, ys, zs, hist, ax, cmap='cubehelix'):
    plot_marginal(ys, zs, xs, hist, ax, 0, 'x', cmap)
    plot_marginal(xs, zs, ys, hist, ax, 1, 'y', cmap)
    plot_marginal(xs, ys, zs, hist, ax, 2, 'z', cmap)


def plot_percentiles(
    vk1r, vk2r, percentiles, hist, samprange, sampres, ax=None, cmap='cubehelix'
):
    if type(cmap) is str:
        cmap = matplotlib.cm.get_cmap(cmap)

    if ax is None:
        ax = pp.gca()

    hnonan = np.where(np.isfinite(hist), hist, np.zeros_like(hist))
    hcumsum = np.cumsum(hnonan, axis=2)
    htotals = np.nanmax(hcumsum, axis=2)[:, :, np.newaxis]
    hvalid_totals = np.bitwise_and(np.isfinite(htotals), htotals != 0)
    hcumsum /= np.where(hvalid_totals, htotals, np.full_like(htotals, np.inf))

    percentiles = np.array(percentiles) / 100
    colors = cmap(percentiles)
    zmx, zmy = np.meshgrid(vk2r, vk1r)

    for perc, color in zip(percentiles, colors):
        vals = np.interp(
            np.array([
                [
                    np.searchsorted(hcumsum[vk1d, vk2d], perc)
                    for vk2d in range(hcumsum.shape[1])
                ]
                for vk1d in range(hcumsum.shape[0])
            ]),
            [0, sampres - 1],
            samprange
        )

        ax.plot_trisurf(
            zmx.flatten(), zmy.flatten(), vals.flatten(),
            color=color, alpha=0.5, label=perc
        )


def make_plots(cached, show_marginals=True, percentiles=(5, 50, 95)):
    """
    TODO: This.

    :param config:
    """

    print 'Plotting marginals.'

    config = cached['config']
    popkey, effectkey = 'h', 'Rhh'

    varkeys, paramkeys = [config['props'][k] for k in 'varkeys', 'paramkeys']
    fig = pp.figure(figsize=(len(varkeys) * 15, len(varkeys) * 10))

    for sampkey in config['args']['samplings']:
        print '\t', sampkey

        for spi, ((vki1, vk1), (vki2, vk2)) in enumerate(
            itertools.combinations_with_replacement(enumerate(varkeys), 2)
        ):
            print '\t\t', vk1, vk2,

            hists = cached['counts'][sampkey][popkey][effectkey]

            ax = fig.add_subplot(
                len(varkeys),
                len(varkeys),
                vki1 * len(varkeys) + vki2 + 1,
                projection='3d'
            )

            vk1r, vk2r = [config['params'][vkn]['range'] for vkn in vk1, vk1]

            sampling = config['args']['samplings'][sampkey]
            sampres = sampling['resolution']
            samprange = sampling['range']

            if len(samprange) > 1:
                hist = np.array(hists[vki1, vki2], dtype=float)
                histf = hist.flatten()

                zs = np.linspace(*(samprange + [sampres]))
                mx, my, mz = np.meshgrid(vk2r, vk1r, zs)
                mzf = mz.flatten()[histf != 0]

                cmap = matplotlib.cm.get_cmap('jet')

                ax.set_xlim(np.amin(vk2r), np.amax(vk2r))
                ax.set_ylim(np.amin(vk1r), np.amax(vk1r))
                ax.set_zlim(np.amin(mzf), np.amax(mzf))
                ax.set_xlabel(vk2)
                ax.set_ylabel(vk1)
                ax.set_zlabel('%s / %s' % (popkey, effectkey))

                if show_marginals:
                    print 'marginals',
                    plot_marginals(vk2r, vk1r, zs, hist, ax, cmap)

                if percentiles is not None and len(percentiles):
                    print 'percentiles',
                    plot_percentiles(
                        vk1r,
                        vk2r,
                        percentiles,
                        hist,
                        samprange,
                        sampres,
                        ax,
                        cmap
                    )

            print

        fn = 'plots/%s-%s.png' % (config['file']['name'], sampkey)
        print '\t\tWriting', fn
        pp.savefig(fn)


def main():
    """Main."""

    make_plots(cPickle.load(open(sys.argv[1])))

if __name__ == '__main__':
    main()
