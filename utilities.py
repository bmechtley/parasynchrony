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
        """
        Default fallback for JSON encoder.

        :param obj: object to encode.
        :return: encoded object.
        """

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
    """
    Simple wrapper to multiprocessing.Pool.map_async to just use regular Python
    map() when the desired number of processes is 1.

    :param mapfun: (callable) Function to map.
    :param mapdata: (list) List of parameter values onto which to map mapfun.
    :param processes: (int) Number of processes to use.
    :param timeout: (float) Maximum computation time for each process.
    :param pargs: (dict) Other arguments to multiprocessing.Pool.map_async.
    :return: (list) list of results of mapfun.
    """

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
    """
    Same as multipool, but instead use itertools.imap and
    multiprocessing.Pool.imap for lazy loading of parameter values.

    :param mapfun: (callable) Function to map.
    :param mapdataiter: (iterator) Iterator of parameter values onto which to
        map mapfun.
    :param processes: (int) Number of processes to use.
    :param timeout: (float) Maximum computation time for each process.
    :param pargs: (dict) Other arguments to multiprocessing.Pool.imap.
    :return: (list) list of results of mapfun.
    """

    if processes == 1:
        return itertools.imap(mapfun, mapdataiter)
    else:
        return multiprocessing.Pool(processes=processes).imap(
            mapfun, mapdataiter, **pargs
        ).get(timeout)


def paramhash(params):
    """
    Create a unique hash for a bunch of parameter values.

    :param params: (dict) dictionary of parameter: value pairs.
    :return: (str) hash string.
    """

    return hash(json.dumps(params, cls=NumpyAwareJSONEncoder))


def dict_merge(a, b):
    """
    Merge two dictionaries, preferring the values from the second in case of
    collision.

    :param a: (dict) first dictionary
    :param b: (dict) second dictionary
    :return: (dict) new dictionary containing keys and values from both
        dictionaries.
    """

    c = a.copy()
    c.update(b)
    return c


def dict_split(d, keys):
    """
    Extract a list of keys into a new dictionary containing only those
    key: value pairs and a dictionary not containing those keys.

    :param d: (dict) original input dictionary
    :param keys: (list) list of key names
    :return: (dict, dict) two dictionaries. The first does not contain the split
        keys, and the second only contains the split keys.
    """

    return {
        k: v for k, v in d.iteritems() if k not in keys
    }, {
        k: v for k, v in d.iteritems() if k in keys
    }


def dict_unpack(d, keys):
    """
    Unpack a dictionary into a list of values given an ordering of keys.

    :param d: (dict) dictionary to unpack
    :param keys: (list) ordered list of keys
    :return: (tuptel) tuple of parameter values, ordered according to keys.
    """

    return tuple([d[k] for k in keys])


def decode_list(data):
    """
    Helper for opening JSON data with non-UTF-8 encoding. This recursively
    converts all elements in a list.

    :param data: (list) the list
    :return: (list) the list with all string elements encoded in UTF-8.
    """

    rv = []

    for item in data:
        if isinstance(item, unicode):
            item = item.encode()
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

    :param data: (dict) the dict
    :return: (dict) the dict with all string elements encoded in UTF-8.
    """

    rv = {}

    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode()
        if isinstance(value, unicode):
            value = value.encode()
        elif isinstance(value, list):
            value = decode_list(value)
        elif isinstance(value, dict):
            value = decode_dict(value)

        rv[key] = value

    return rv


def zero_storage_arrays(config):
    """
    Create a bunch of arrays of zeros for large marginal runs given a
    configuration dict. See marginals.py or slices.py for more information on
    how these config dictionariesare formatted.

    :param config: (dict) configuration dict.
    :return: (dict) dictionary containing:
        counts: (np.array) storage of histogram
        maxima: (np.array) storage of maxima
        samples: (np.array) storage of random samples
        samplesleft: (np.array) storage of how many random samples are left
            to compute
    """

    # TODO: Which of these is actually used?? This or the one in marginals.py?

    paramkeys = config['props']['paramkeys']
    nparams = len(paramkeys)
    paramres, samplings = [config['args'][k] for k in 'resolution', 'samplings']

    # TODO: This is messy. Ideally, I'd be able to change which metrics are
    #   returned in compute_metrics and have the structure of these
    #   histograms automatically change. Easy fix is to wait to create
    #   the matrices until we see the first dict of values returned.
    popkeys, effectkeys = ('h', 'p'), ('Rhh', 'Rpp')

    # Parameters that actually vary, their count, and their index within the
    # ordered parameter list.
    varkeys = config['props']['varkeys']
    nvarkeys = len(varkeys)

    # Construct the statistic matrices.
    # varindex1, varindex2, paramindex1, paramindex2, ...
    histshape = (nvarkeys, nvarkeys, paramres, paramres)

    counts, maxima, samples, samplesleft = [
        {
            sampkey: {
                popkey: {
                    effectkey: None for effectkey in effectkeys
                }
                for popkey in popkeys
            }
            for sampkey in samplings
        }
        for _ in range(4)
    ]

    for popkey in popkeys:
        for effectkey in effectkeys:
            for sampkey, sampling in samplings.iteritems():
                nsamples = int(sampling['nsamples'])
                sampres = sampling['resolution']

                counts[sampkey][popkey][effectkey] = np.zeros(
                    histshape + (sampres,), dtype=int
                )

                # Store the list of argmax parameter values + the maximum metric
                # value.
                maxima[sampkey][popkey][effectkey] = np.zeros(
                    histshape + (nparams + 1,), dtype=float
                )

                # Random samples are for the entire hypercube and not for each
                # marginal. Store the list of parameter values + the metric
                # value.
                samples[sampkey][popkey][effectkey] = np.zeros(
                    (nsamples, nparams + 1), dtype=float
                )

                # How many samples we have left to compute. Decrements. Note
                # that this may not actually reach zero, so it'll be important
                # to check it when plotting. Samples will start from the end,
                # as they are placed at samplesleft.
                samplesleft[sampkey][popkey][effectkey] = nsamples

    return dict(
        counts=counts,
        maxima=maxima,
        samples=samples,
        samplesleft=samplesleft
    )


def load_config(configfile):
    """
    Load a configuration JSON file using pybatchdict, decoding UTF-8.

    :param configfile: (str) path to configuration file.
    :return: (dict) configuration dictionary.
    """

    config_json = json.load(
        open(configfile),
        object_hook=decode_dict
    )

    return pybatchdict.BatchDict(config_json)


def noise_cov(ndict):
    """
    Noise covariance from parameterization dictionary.

    :param ndict: (dict) dictionary of noise parameters including:
        SpSh: Ratio of parasitoid variance to host variance (Sp / Sh).
        Chh: Correlation between hosts.
        Cpp: Correlation between parasitoids.
    :return: (np.ndarray) 4x4 covariance matrix.
    """

    return np.array([
        [1, ndict['Chh'], 0, 0],
        [ndict['Chh'], 1, 0, 0],
        [0, 0, ndict['SpSh'], ndict['SpSh'] * ndict['Cpp']],
        [0, 0, ndict['SpSh'] * ndict['Cpp'], ndict['SpSh']]
    ])


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

    :param shape: (int or tuple of ints) shape to normalize.
    :return: (tuple) normalized shape tuple.
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

    :param a: (np.array) an n-dimensional numpy array
    :param ws: (int or tuple of ints) int or tuple of ints representing the size
        of each dimension of the window.
    :param ss: (int or tuple of ints) int or tuple of ints representing the
        amount to slide the window in each dimension. If not specified, it
        defaults to ws.
    :param flatten: (bool) If True, all slices are flattened. Otherwise, there
        is an extra dimension for each dimension of the input.
    :return: (np.array) an array containing each n-dimensional window from a.
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

    # the strides tuple will be the array's strides multiplied by step size,
    # plus the array's strides (tuple addition)
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
