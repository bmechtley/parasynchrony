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

def plot_marginals(
    config,
    show_marginals=True,
    show_bins=False,
    percentiles=(5, 50, 95)
):
    """
    TODO: This.

    :param config:
    """

    print 'Plotting marginals.'

    cacheprefix = os.path.join(config['file']['dir'], config['file']['name'])
    gathered = cPickle.load(open('%s-full.pickle' % cacheprefix))

    popkey, effectkey = 'h', 'Rhh'
    hists = gathered['counts'][popkey][effectkey]

    varkeys, paramkeys = [config['props'][k] for k in 'varkeys', 'paramkeys']
    fig = pp.figure(figsize=(len(varkeys) * 15, len(varkeys) * 10))

    for sampkey in config['samplings']:
        for spi, ((vki1, vk1), (vki2, vk2)) in enumerate(
            itertools.combinations_with_replacement(enumerate(varkeys), 2)
        ):
            ax = fig.add_subplot(
                len(varkeys),
                len(varkeys),
                vki1 * len(varkeys) + vki2 + 1,
                projection='3d'
            )

            vk1r, vk2r = [config['params'][vkn]['range'] for vkn in vk1, vk1]

            sampling = config['args']['samplings'][sampkey]
            samprange = sampling['range']
            sampres = sampling['resolution']

            hist = np.array(hists[sampkey][vki1, vki2], dtype=float)
            histf = hist.flatten()

            zs = np.linspace(float(samprange[0]), float(samprange[1]), sampres)

            mx, my, mz = np.meshgrid(vk2r, vk1r, zs)
            mxf, myf, mzf = [a.flatten() for a in mx, my, mz]

            pzs = np.tile(
                np.array([
                    scipy.stats.percentileofscore(histf, score) for score in zs
                ])[np.newaxis, np.newaxis, :],
                (len(vk2r), len(vk1r), 1)
            )
            pzsf = pzs.flatten()

            mxf, myf, mzf, pzsf, histf = [
                a[histf != 0] for a in mxf, myf, mzf, pzsf, histf
            ]

            cmap = matplotlib.cm.get_cmap('jet')
            maxcount = float(np.amax(hist))

            ax.set_xlim(np.amin(vk2r), np.amax(vk2r))
            ax.set_ylim(np.amin(vk1r), np.amax(vk1r))
            ax.set_zlim(np.amin(mzf), np.amax(mzf))
            ax.set_xlabel(vk2)
            ax.set_ylabel(vk1)
            ax.set_zlabel('%s / %s' % (popkey, effectkey))

            if show_marginals:
                xmy, xmz = np.meshgrid(vk1r, zs)
                xmx = np.sum(hist, axis=0).T / np.amax(hist)
                ax.contourf(
                    xmx, xmy, xmz,
                    zdir='x',
                    offset=vk2r[0] - np.diff(vk2r)[0] / 10.0,
                    cmap=cmap,
                    alpha=0.5
                )

                ymx, ymz = np.meshgrid(vk2r, zs)
                ymy = np.sum(hist, axis=1).T / np.amax(hist)
                ax.contourf(
                    ymx, ymy, ymz,
                    zdir='y',
                    offset=vk1r[-1] + np.diff(vk1r)[-1] / 10.0,
                    cmap=cmap,
                    alpha=0.5
                )

                zmx, zmy = np.meshgrid(vk2r, vk1r)
                zmz = np.sum(hist, axis=2) / np.amax(hist)
                ax.contourf(
                    zmx, zmy, zmz,
                    zdir='z',
                    offset=zs[0] - np.diff(zs)[0] / 10.0,
                    cmap=cmap,
                    alpha=0.5
                )

            if show_bins:
                ax.scatter(
                    mxf, myf, mzf,
                    s=np.log10(1 + histf / maxcount) * 1000,
                    c=cmap(pzsf / np.amax(pzsf)),
                    lw=0,
                    alpha=1.0
                )

            cumsums = np.cumsum(hist, axis=2)
            cumsums /= np.amax(cumsums, axis=2)[:, :, np.newaxis]

            if percentiles is not None and len(percentiles):
                percentiles = np.array(percentiles) / 100
                colors = cmap(percentiles)

                for perc, color in zip(percentiles, colors):
                    vals = np.interp(
                        np.array([
                            [
                                np.searchsorted(cumsums[vk1d, vk2d], perc)
                                for vk2d in range(cumsums.shape[1])
                            ]
                            for vk1d in range(cumsums.shape[0])
                        ]),
                        [0, sampling['resolution'] - 1],
                        sampling['range']
                    )

                    ax.plot_trisurf(
                        zmx.flatten(),
                        zmy.flatten(),
                        vals.flatten(),
                        color=color,
                        alpha=0.5,
                        label=perc
                    )

                ax.legend()

        pp.savefig('%s-%s.png' % (cacheprefix, sampkey))

def main():
    """Main."""

    plot_marginals(utilities.config_defaults(sys.argv[1]))

if __name__ == '__main__':
    main()
