"""
utilities.py
parasynchrony
2014 Brandon Mechtley

Various utility helper functions shared between the different analysis scripts.
"""

import json
import pybatchdict


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
