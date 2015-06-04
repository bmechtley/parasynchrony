"""
utilities.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Various utility helper functions shared between the different analysis scripts.
"""

import json
import os.path
import operator
import itertools
import functools
import collections
import multiprocessing

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

import pybatchdict


class NumpyAwareJSONEncoder(json.JSONEncoder):
    """
    https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)

def multipool(
        mapfun,
        mapdata,
        processes=multiprocessing.cpu_count(),
        timeout=99999,
        **pargs
):
    if processes == 1:
        return map(mapfun, mapdata)
    else:
        return multiprocessing.Pool(processes=processes).map_async(
            mapfun, mapdata, **pargs
        ).get(timeout)


def imultipool(
        mapfun,
        mapdataiter,
        processes=multiprocessing.cpu_count(),
        timeout=99999,
        **pargs
):
    if processes == 1:
        return itertools.imap(mapfun, mapdataiter)
    else:
        return multiprocessing.Pool(processes=processes).imap(
            mapfun, mapdataiter, **pargs
        ).get(timeout)


def paramhash(params):
    return hash(json.dumps(params, cls=NumpyAwareJSONEncoder))


def dict_merge(a, b):
    """
    Merge two dictionaries, preferring the values from the second in case of
    collision.

    :param a: first dictionary
    :param b: second dictionary
    :return: new dictionary containing keys and values from both dictionaries.
    """

    c = a.copy()
    c.update(b)
    return c


def dict_split(d, keys):
    return {
        k: v for k, v in d.iteritems() if k not in keys
    }, {
        k: v for k, v in d.iteritems() if k in keys
    }


def dict_unpack(d, keys):
    return tuple([d[k] for k in keys])


def decode_list(data):
    """
    Helper for opening JSON data with non-UTF-8 encoding. This recursively
    converts all elements in a list.
    :param data: the list
    :return: the list with all string elements encoded in UTF-8.
    """

    rv = []

    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = decode_list(item)
        elif isinstance(item, dict):
            item = decode_dict(item)

        rv.append(item)

    return rv


def decode_dict(data):
    """
    Helper for opening JSON data with non-UTF-8 encoding. This recursively
    converts all elements in a dict.
    :param data: the dict
    :return: the dict with all string elements encoded in UTF-8.
    """

    rv = {}

    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = decode_list(value)
        elif isinstance(value, dict):
            value = decode_dict(value)

        rv[key] = value

    return rv

def zero_storage_arrays(config):
    """

    :param config:
    :return:
    """

    paramkeys = config['props']['paramkeys']
    nparams = len(paramkeys)

    nruns = functools.reduce(
        operator.mul,
        [len(param) for param in config['params'].values()],
        1
    )

    res, samplings = [config['args'][k] for k in 'resolution', 'samplings']

    # TODO: This is messy. Ideally, I'd be able to change which metrics are
    # returned in compute_metrics and have the structure of these histograms
    # automatically change. Easy fix is to wait to create the matrices until we
    # see the first dict of values returned.
    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')

    # Parameters that actually vary, their count, and their index within the
    # ordered parameter list.
    varkeys = config['props']['varkeys']
    nvarkeys = len(varkeys)

    # Construct the statistic matrices.
    # varindex1, varindex2, paramindex1, paramindex2, ...
    histshape = (nvarkeys, nvarkeys, res, res)

    counts, maxima, samples, samplesleft = [
        {
            popkey: {
                effectkey: {
                    sampkey: None
                    for sampkey in samplings
                } for effectkey in effectkeys
            } for popkey in popkeys
        }
        for _ in range(4)
    ]

    for popkey in popkeys:
        for effectkey in effectkeys:
            for sampkey, sampling in samplings.iteritems():
                nsamples = int(sampling['nsamples'])

                counts[popkey][effectkey][sampkey] = np.zeros(
                    histshape + (sampling['resolution'],), dtype=int
                )

                # Store the list of argmax parameter values + the maximum metric
                # value.
                maxima[popkey][effectkey][sampkey] = np.zeros(
                    histshape + (nparams + 1,), dtype=float
                )

                # Random samples are for the entire hypercube and not for each
                # marginal. Store the list of parameter values + the metric
                # value.
                samples[popkey][effectkey][sampkey] = np.zeros(
                    (nsamples, nparams + 1), dtype=float
                )

                # How many samples we have left to compute. Decrements. Note
                # that this may not actually reach zero, so it'll be important
                # to check it when plotting. Samples will start from the end,
                # as they are placed at samplesleft.
                samplesleft[popkey][effectkey][sampkey] = nsamples

    return dict(
        counts=counts,
        maxima=maxima,
        samples=samples,
        samplesleft=samplesleft
    )

def load_config(configfile):
    config_json = json.load(
        open(configfile, 'r'),
        object_hook=decode_dict
    )

    return pybatchdict.BatchDict(config_json)

def config_defaults(configpath=None):
    """
    Open a configuration file and reformat it.

    :param configpath: path to configuration JSON file.
    :return: (dict) Configuration dictionary with parameter ranges replaced with
        lists of parameter values and a few extra keys:
        file:
            dir: directory to store cached files.
            name: filename prefix for cached files.
            slice_size: number of computations per individual run.
        props:
            paramkeys: ordered list of parameter names.
            varkeys: ordered list of parameter names for those that vary.
    """

    if configpath is not None:
        configdir, configfile = os.path.split(configpath)
        configname = os.path.splitext(configfile)[0]
        config = json.load(open(configpath))
        config.setdefault('file', dict())
        config['file'].setdefault('dir', configdir)
        config['file'].setdefault('name', configname)
        config['file'].setdefault('slice_size', 20)
    else:
        config = dict(
            file=dict(
                dir='cache/',
                name='fracsync-marginals-default',
                slice_size=20
            ),
            args=dict(
                resolution=10,
                samplings=dict(
                    zero_one=dict(
                        range=[0, 1], res=100, inclmin=True, nsamples=0
                    ),
                    one_ten=dict(
                        range=[1, 10], res=100, inclmin=False, nsamples=1000
                    ),
                    gt_ten=dict(
                        range=[10], res=1, inclmin=False, nsamples=1000
                    )
                )
            ),
            params=dict(
                r=dict(default=2.0, range=[1.1, 4]),
                a=dict(default=1.0),
                c=dict(default=1.0),
                k=dict(default=0.5, range=[0.1, 0.9]),
                mh=dict(default=0.25, range=[0, 0.5]),
                mp=dict(default=0.25, range=[0, 0.5]),
                SpSh=dict(default=1.0, range=[0, 10]),
                Chh=dict(default=0.5, range=[0, 1]),
                Cpp=dict(default=0.5, range=[0, 1])
            )
        )

    # Make sure the params are ordered consistently so we can easily slice
    # combinations without having to actually store info regarding to which
    # parameters each computation should use.
    config['params'] = collections.OrderedDict(config['params'])
    res = config['args']['resolution']

    config['props'] = dict(varkeys=[])

    for p in config['params'].itervalues():
        p.setdefault('resolution', res)
        p.setdefault('range', (p['default'], p['default']))

        if p['range'][0] != p['range'][1]:
            p['range'] = np.linspace(p['range'][0], p['range'][1], res)
        else:
            p['range'] = [p['default']]

    config['props'] = dict(paramkeys=config['params'].keys())
    config['props']['varkeys'] = [
        k for k in config['props']['paramkeys']
        if len(config['params'][k]['range']) > 1
    ]

    return config

def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple, even
    for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    """

    try:
        i = int(shape)
        return i,
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    """
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    """

    if ss is None:
        ss = ws

    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in
    # every dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if len(set(ls)) != 1:
        raise ValueError(
            'a.shape, ws and ss must all have the same length.'
            'They were %s' % str(ls)
        )

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(
            'ws cannot be larger than a in any dimension. '
            'a.shape was %s and ws was %s' % (str(a.shape), str(ws))
        )

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    print 'newshape', newshape

    # the shape of the strided array will be the number of slices in each
    # dimension plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)

    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)

    if not flatten:
        return strided
    else:
        # Collapse strided so that it has one more dimension than the window.
        # i.e. the new array is a flat list of slices.
        meat = len(ws) if ws.shape else 0
        firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
        dim = firstdim + (newshape[-meat:])

        # remove any dimensions with size 1
        dim = filter(lambda i: i != 1, dim)
        return strided.reshape(dim)
