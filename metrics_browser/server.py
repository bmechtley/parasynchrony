import itertools
import StringIO
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from PIL import Image

import numpy as np
import matplotlib.cm
import matplotlib.colors

import flask

from lru import LRU

import models


app = flask.Flask(__name__)

model = models.Parasitism.get_model("nbd(2)")
covariance = np.array([
    [1E-1, 1E-2, 0, 0],
    [1E-2, 1E-2, 0, 0],
    [0, 0, 1E-1, 1E-2],
    [0, 0, 1E-2, 1E-1]
])

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


@app.route('/pcolor.json')
def get_pcolor():
    params = {k: flask.request.args.get(k) for k in [
        'r', 'a', 'c', 'k', 'mh', 'mp'
    ]}
    axes = {k: flask.request.args.get(k) for k in ['axis_x', 'axis_y']}
    dims = {k: flask.request.args.get(k) for k in ['width', 'height']}

    hhash = str(hash(frozenset(
        sorted(params.items()) + sorted(axes.items()) + sorted(dims.items())
    )))

    print "<pcolor> Loading transfer function %s." % hhash

    if hhash not in hcache.keys():
        print "<pcolor>\tComputing new."
        sym_params = {
            models.Parasitism.params[k]: v for k, v in params.iteritems()
        }
        xfer = model.transfer_function(sym_params)

        valrange_x = np.linspace(-1, 1, dims.get('width', 100))
        valrange_y = np.linspace(-1, 1, dims.get('height', 100))

        h = np.array([
            [
                xfer(complex(re, im))
                for re in valrange_x
            ]
            for im in valrange_y
        ])

        logmagh = np.log(np.abs(h) + np.finfo(h.dtype).eps)

        color_norm = matplotlib.colors.Normalize(
            vmin=np.amin(logmagh[logmagh != np.amin(logmagh)]),
            vmax=np.amax(logmagh)
        )

        pcolor = matplotlib.cm.cubehelix(color_norm(logmagh))

        hcache[hhash] = dict(
            images=np.array([
                [
                    Image.fromarray(np.uint8(pcolor[:, :, row, col] * 255))
                    for col in range(h.shape[3])
                ]
                for row in range(h.shape[2])
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
