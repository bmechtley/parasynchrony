import itertools
import StringIO
import json
import sys
import os
import PIL
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import matplotlib.cm
import matplotlib.colors

import flask
from lru import LRU

import models

app = flask.Flask(__name__)
model = models.parasitism.get_model("nbd(2)")
hcache = LRU(100)


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/image/<hhash>_<int:row>_<int:col>.png')
def get_image(hhash, row, col):
    print "<image> Looking for image %s (%d, %d)." % (hhash, row, col)

    if hhash in hcache.keys():
        print "<image>\tFound!"
        img_io = StringIO.StringIO()
        hcache[hhash]['images'][row, col].save(img_io, 'PNG')
        img_io.seek(0)
        return flask.send_file(img_io, mimetype='image/png')
    else:
        print "<image>\tNot found :("
        flask.abort(404)


def transfer_function(params=None, domain=None, **_):
    sym_params = {
        models.parasitism.symbols[k]: v for k, v in params.iteritems()
    }

    xfer = model.transfer_function(sym_params)

    h = np.array([
        [
            xfer(complex(re, im))
            for re in domain[1]
        ]
        for im in domain[0]
    ])

    logmagh = np.log(np.abs(h) + np.finfo(h.dtype).eps)

    color_norm = matplotlib.colors.Normalize(
        vmin=np.amin(logmagh[logmagh != np.amin(logmagh)]),
        vmax=np.amax(logmagh)
    )

    pcolor = matplotlib.cm.cubehelix(color_norm(logmagh))

    return pcolor


def fraction_synchrony(
        params=None,
        domain=None,
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

    fracfuns = dict(avg=np.mean, max=np.amax)

    results = {
        k: np.zeros((
            len(domain[0]),
            len(domain[1]),
            len(model.vars),
            len(model.vars)
        ))
        for k in fracfuns
    }

    variants = {key: dict(
        params=dict(params.iteritems()),
        noise=dict(noise.iteritems())
    ) for key in ['host', 'nohost']}

    variants['nohost']['params']['mh'] = 0
    variants['nohost']['noise']['Shh'] = 0

    for (i, var1), (j, var2) in itertools.product(
        enumerate(domain[0]), enumerate(domain[1])
    ):
        for name, variant in variants.iteritems():
            for k, v in itertools.izip(
                    [ax['param'] for ax in axes], [var1, var2]
            ):
                if k in variant['params']:
                    variant['params'][k] = v
                if k in variant['noise']:
                    variant['noise'][k] = v

            if name == 'nohost':
                variant['params']['mh'] = 0
                variant['noise']['Shh'] = 0

            sym_params = {
                models.parasitism.params[k]: v
                for k, v in variant['params'].iteritems()
            }

            noisecov = np.array([
                [variant['noise']['Sh'], variant['noise']['Shh'], 0, 0],
                [variant['noise']['Shh'], variant['noise']['Sh'], 0, 0],
                [0, 0, variant['noise']['Sp'], variant['noise']['Spp']],
                [0, 0, variant['noise']['Spp'], variant['noise']['Sp']]
            ])

            variant['spectrum'] = np.abs(np.array([model.calculate_spectrum(
                sym_params, noisecov, freq
            ) for freq in np.linspace(0, 0.5, nfreqs)]))

            variant['covariance'] = model.calculate_covariance(
                sym_params, noisecov
            )

            variant['normed'] = np.rollaxis(np.array([
                [
                    np.real(variant['spectrum'][:, row, col]) / np.sqrt(
                        np.prod([
                            variant['covariance'][c, c]**2 for c in [row, col]
                        ])
                    ) for col in range(variant['covariance'].shape[1])
                ] for row in range(variant['covariance'].shape[0])
            ]), 2, 0)

        for key, fracfun in fracfuns.iteritems():
            results[key][i, j, :, :] = np.log(fracfun(
                np.divide(
                    variants['host']['normed'] - variants['nohost']['normed'],
                    variants['nohost']['normed']
                ),
                axis=0
            ))

    color_norms = {
        k: matplotlib.colors.Normalize(
            vmin=np.amin(result), vmax=np.amax(result)
        ) for k, result in results.iteritems()
    }

    return {
        k: np.array([
            [
                matplotlib.cm.cubehelix(color_norms[k](result[:, :, col, row]))
                for col in range(result.shape[3])
            ] for row in range(result.shape[2])
        ]).swapaxes(0, 2).swapaxes(1, 3)
        for k, result in results.iteritems()
    }


@app.route('/pcolor.json')
def get_pcolor():
    request = flask.request.args

    inputs = dict(
        params={
            k: float(request.get(k))
            for k in ['r', 'a', 'c', 'k', 'mh', 'mp']
        },
        axes=[
            dict(
                param=request.get('axis_%s' % ax),
                min=float(request.get('axis_%s_min' % ax)),
                max=float(request.get('axis_%s_max' % ax)),
                n=int(request.get('axis_%s_n' % ax))
            ) for ax in ['x', 'y']
        ],
        noise={k: float(request.get(k)) for k in ['Shh', 'Sh', 'Spp', 'Sp']}
    )

    hhash = str(hash(json.dumps(inputs, sort_keys=True)))

    inputs['domain'] = [
        np.linspace(ax['min'], ax['max'], ax['n'])
        for ax in inputs['axes'][::-1]
    ]

    metric_name = request.get('metric')
    metric = dict(
        xfer=transfer_function,
        avgsync=lambda **kw: fraction_synchrony(**kw)['avg'],
        maxsync=lambda **kw: fraction_synchrony(**kw)['max']
    )[metric_name]

    print "<pcolor> Loading metric %s, %s." % (metric_name, hhash), inputs

    if hhash not in hcache.keys():
        print "<pcolor>\tComputing new."

        pcolor = metric(**inputs)
        print 'pcolor shape', pcolor.shape

        hcache[hhash] = dict(
            images=np.array([
                [
                    PIL.Image.fromarray(
                        np.uint8(pcolor[:, :, row, col, :].swapaxes(0, 1) * 255)
                    )
                    for col in range(pcolor.shape[3])
                ]
                for row in range(pcolor.shape[2])
            ], dtype=object)
        )

    jsondata = dict(
        hash=hhash,
        images=[
            [
                '/image/%s_%d_%d.png' % (hhash, row, col)
                for col in range(hcache[hhash]['images'].shape[1])
            ]
            for row in range(hcache[hhash]['images'].shape[0])
        ]
    )

    return flask.jsonify(**jsondata)

if __name__ == '__main__':
    app.run(debug=True)
