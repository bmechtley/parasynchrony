"""
utilities.py
parasynchrony
2014 Brandon Mechtley
Reuman Lab, Kansas Biological Survey

Various utility helper functions shared between the different analysis scripts.
"""

import json
import pybatchdict
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast


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


def load_config(configfile):
    config_json = json.load(
        open(configfile, 'r'),
        object_hook=decode_dict
    )

    return pybatchdict.BatchDict(config_json)


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
