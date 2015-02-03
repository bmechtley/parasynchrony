import itertools
import StringIO
import json
import sys
import os

import numpy as np

import flask
from lru import LRU

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import models

app = flask.Flask(__name__)
model = models.parasitism.get_model("nbd(2)")
datacache = LRU(100)


@app.route('/')
def index():
    return flask.render_template('index.html')


def fraction_synchrony(
    params=None,
    axes=None,
    noise=None,
    nfreqs=100,
    **_
):
    """
    Compute the fraction of synchrony between patches for which the host is
    responsible using different metrics (fraction of spectrum averages,
    fraction of spectrum maximums).

    TODO: Allow caching of spectra.
    TODO: Return fraction for which different populations are responsible.
    TODO: Optimize.
    TODO: Allow for different parameter ranges.

    :param params: model parameter dictionary.
    :param dims: number of datapoints along each dimension.
    :param axes: over which parameters to iterate.
    :param noise: noise parameters (H/P variance, H-H/P-P covariance)
    :param resolution: number of frequency values to compute.
    :return: dictionary of metric: fraction pairs.
    """

    fracfuns = dict(avgfracsync=np.mean, maxfracsync=np.amax)

    results = {
        k: np.zeros((
            len(axes['y']['domain']),
            len(axes['x']['domain']),
            len(model.vars),
            len(model.vars)
        ))
        for k in fracfuns
    }

    freqs = np.linspace(0, 0.5, nfreqs)

    for (i, vary), (j, varx) in itertools.product(*[
        enumerate(np.linspace(axes[ax]['min'], axes[ax]['max'], axes[ax]['n']))
        for ax in ['y', 'x']
    ]):
        correlations = dict()

        for name in ['all', 'hostonly']:
            vparams, vnoise = dict(params.items()), dict(noise.items())

            for k, v in itertools.izip(
                [axes[ax]['param'] for ax in ['y', 'x']], [vary, varx]
            ):
                if k in vparams:
                    vparams[k] = v
                elif k in vnoise:
                    vnoise[k] = v

            if name == 'hostonly':
                vparams['mp'] = 0
                vnoise['Spp'] = 0
                vnoise['Sp'] = 0

            sym_params = models.parasitism.sym_params(vparams)

            noisecov = np.array([
                [vnoise['Sh'], vnoise['Shh'], 0, 0],
                [vnoise['Shh'], vnoise['Sh'], 0, 0],
                [0, 0, vnoise['Sp'], vnoise['Spp']],
                [0, 0, vnoise['Spp'], vnoise['Sp']]
            ])

            spectrum = np.abs(np.array([model.calculate_spectrum(
                sym_params, noisecov, freq
            ) for freq in freqs]))

            cov = model.calculate_covariance(sym_params, noisecov)

            # Compute normalized spectrum, i.e. correlation matrix.
            correlations[name] = np.rollaxis(np.array([
                [
                    np.divide(
                        np.real(spectrum[:, row, col]),
                        np.sqrt(np.prod([cov[c, c]**2 for c in [row, col]]))
                    ) for col in range(cov.shape[1])
                ] for row in range(cov.shape[0])
            ]), 2, 0)

        for key, fracfun in fracfuns.iteritems():
            results[key][i, j, :, :] = fracfun(
                correlations['hostonly'] / correlations['all'],
                axis=0
            )

    return results


@app.route('/pcolor.json')
def get_pcolor():
    request = flask.request.args

    inputs = dict(
        params={
            k: float(request.get(k))
            for k in ['r', 'a', 'c', 'k', 'mh', 'mp']
        },
        axes={
            ax: dict(
                param=request.get('axis_%s' % ax),
                min=float(request.get('axis_%s_min' % ax)),
                max=float(request.get('axis_%s_max' % ax)),
                n=int(request.get('axis_%s_n' % ax))
            ) for ax in ['x', 'y']
        },
        noise={k: float(request.get(k)) for k in ['Shh', 'Sh', 'Spp', 'Sp']}
    )

    # List of parameters that are used as axes.
    axes_params = [ax['param'] for ax in inputs['axes'].values()]

    # Take the axes out of the params to avoid duplicate calculations in
    # multiple hashes.
    for ax in axes_params:
        inputs['params'][ax] = 0

    # Hash the input parameters for caching calculations.
    paramhash = str(hash(json.dumps(inputs, sort_keys=True)))
    metric_name = request.get('metric')

    print "<pcolor> Loading metric %s, %s." % (metric_name, paramhash), inputs

    if paramhash not in datacache.keys():
        print "<pcolor>\tComputing new."

        for ax in inputs['axes'].values():
            ax['domain'] = list(np.linspace(ax['min'], ax['max'], ax['n']))

        datacache[paramhash] = dict(
            data=fraction_synchrony(**inputs),
            inputs=inputs
        )

    cached = datacache[paramhash]
    shape = cached['data'][metric_name].shape

    jsondata = dict(
        hash=paramhash,
        data=dict(children=[
            dict(
                vary=str(model.vars[prow]),
                varx=str(model.vars[pcol]),
                children=[dict(
                    x=cached['inputs']['axes']['x']['domain'][vcol],
                    y=cached['inputs']['axes']['y']['domain'][vrow],
                    z=cached['data'][metric_name][vrow, vcol, prow, pcol]
                ) for vrow, vcol in itertools.product(
                    range(shape[0]), range(shape[1])
                )]
            ) for prow, pcol in itertools.product(
                range(shape[2]), range(shape[3])
            )
        ]),
        inputs=inputs
    )

    return flask.jsonify(**jsondata)

if __name__ == '__main__':
    app.run(debug=True)
