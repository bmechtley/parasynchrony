import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pp

import itertools
import models


class TransferPlot:
    def __init__(self):
        self.model = models.Parasitism.get_model("nbd(2)")
        self.nvars = len(self.model.vars)
        self.covariance = np.array([
            [1E-1, 1E-2, 0, 0],
            [1E-2, 1E-2, 0, 0],
            [0, 0, 1E-1, 1E-2],
            [0, 0, 1E-2, 1E-1]
        ])

        self.showz, self.showeigs, self.modelstr = True, True, "nbd(2)"

        self.nvals = 20
        self.values = np.linspace(-1.0, 1.0, self.nvals)
        self.extent = [min(self.values), max(self.values)] * 2
        self.circlepts = [
            np.exp(1j * f) for f in np.linspace(0, 2 * np.pi, 100)
        ]

        self.fig, self. subplots = pp.subplots(self.nvars, self.nvars)
        pp.subplots_adjust(top=0.78, bottom=0.1, wspace=0.3, hspace=0.3)

        self.imgs = np.array(
            [[None for c in range(self.nvars)] for r in range(self.nvars)],
            dtype=object
        )

        self.scats = np.array(
            [[None for c in range(self.nvars)] for r in range(self.nvars)],
            dtype=object
        )

        self.showeigs = True
        self.showz = True

        self.sliders = dict(
            r=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.96, 0.8, 0.01], axisbg='white'),
                '$\\lambda$', 0, 10.0, valinit=3.0
            ),
            a=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.94, 0.8, 0.01], axisbg='white'),
                '$a$', 0, 10.0, valinit=0.5
            ),
            c=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.92, 0.8, 0.01], axisbg='white'),
                '$c$', 0, 10.0, valinit=1.0
            ),
            k=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.90, 0.8, 0.01], axisbg='white'),
                '$k$', 0, 10.0, valinit=0.9
            ),
            mh=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.88, 0.8, 0.01], axisbg='white'),
                '$m_H$', 0, 0.5, valinit=0.1
            ),
            mp=matplotlib.widgets.Slider(
                pp.axes([0.1, 0.86, 0.8, 0.01], axisbg='white'),
                '$m_P$', 0, 0.5, valinit=0.1
            )
        )

        self.toggles = dict(
            modelb=matplotlib.widgets.Button(
                pp.axes([0.1, 0.81, 0.08, 0.04], axisbg='white'), 'model'
            ),
            eigb=matplotlib.widgets.Button(
                pp.axes([0.2, 0.81, 0.08, 0.04], axisbg='white'),
                'show $\\lambda$'
            ),
            zb=matplotlib.widgets.Button(
                pp.axes([0.3, 0.81, 0.08, 0.04], axisbg='white'), 'show $Z$'
            )
        )

        self.toggles['modelb'].on_clicked(self.toggle_model)
        self.toggles['eigb'].on_clicked(self.toggle_showz)
        self.toggles['zb'].on_clicked(self.toggle_showeigs)

        self.h = np.zeros([self.nvals, self.nvals, self.nvars, self.nvars])

        for skey in self.sliders:
            self.sliders[skey].on_changed(self.update)

        self.update(None)

    def toggle_showz(self):
        self.showz = not self.showz
        self.update(None)

    def toggle_showeigs(self):
        self.showeigs = not self.showeigs
        self.update(None)

    def toggle_model(self):
        pass

    def update(self, val):
        params = {k: self.sliders[k].val for k in self.sliders}
        sym_params = {
            models.Parasitism.params[k]: v for k, v in params.iteritems()
        }
        xfer = self.model.transfer_function(sym_params)

        valrange = range(self.nvals)
        varrange = range(self.nvars)

        # Plot Z transform.
        if self.showz:
            for iim in valrange:
                for ire in valrange:
                    self.h[iim, ire] = np.log(
                        np.abs(
                            xfer(complex(self.values[ire], self.values[iim]))
                        ) + np.finfo(self.h.dtype).eps
                    )

            for row, col in itertools.product(varrange, repeat=2):
                if self.imgs[row, col] is None:
                    self.imgs[row, col] = self.subplots[row, col].imshow(
                        self.h[:, :, row, col],
                        extent=self.extent,
                        cmap='cubehelix'
                    )
                    self.subplots[row, col].plot(
                        np.real(self.circlepts), np.imag(self.circlepts),
                        color='white', ls=':', alpha=0.5
                    )
                    self.subplots[row, col].axhline(
                        0, color='white', ls=':', alpha=0.5
                    )
                    self.subplots[row, col].axvline(
                        0, color='white', ls=':', alpha=0.5
                    )
                    self.subplots[row, col].set_xlabel('$Re(z)$')
                    self.subplots[row, col].set_ylabel('$Im(z)$')
                else:
                    self.imgs[row, col].set_data(self.h[:, :, row, col])

        # Plot eigenvalues.
        if self.showeigs:
            eigs = self.model.calculate_eigenvalues(sym_params)[0]

            for row, col in itertools.product(varrange, repeat=2):
                if self.scats[row, col] is None:
                    self.scats[row, col] = self.subplots[row, col].scatter(
                        np.real(eigs), np.imag(eigs), color='white', marker='.'
                    )
                else:
                    self.scats[row, col].set_offsets(
                        np.array([np.real(eigs), np.imag(eigs)]).T
                    )

        for row, col in itertools.product(varrange, repeat=2):
            self.subplots[row, col].set_xlim(-1, 1)
            self.subplots[row, col].set_ylim(-1, 1)

        self.fig.canvas.draw_idle()

plot = TransferPlot()
pp.show()
