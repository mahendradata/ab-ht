# --------------------------------------------------------------
# Author: Mahendra Data - mahendra.data@ub.ac.id
# License: BSD 3 clause
# --------------------------------------------------------------

import json
import pprint


class Dict(dict):
    """Dictionary Class
    
    A Dictionary class which its attributes can be accessed using dot.notation.
    
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Configuration(object):
    """JSON formated configuration
    
    Read JSON formated configuration. Its attributes can be accessed using dot.notation.

    Examples
    --------

    >>> from eil.configuration import Configuration
    
    >>> conf = Configuration.load({
    ...         "split_criterion": 'gini',
    ...         "split_confidence": 1e-5,
    ...         "grace_period": 200000,
    ...         "n_models": 5,
    ...         "seed": 42
    ...         })
    >>> print(conf.split_criterion)
    gini

    """

    @staticmethod
    def load(data):
        if type(data) is dict:
            return Configuration.__load_dict__(data)
        elif type(data) is list:
            return [Configuration.load(item) for item in data]
        else:
            return data

    @staticmethod
    def __load_dict__(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Configuration.load(value)
        return result

    @staticmethod
    def read_json(path: str):
        with open(path, "r") as f:
            result = Configuration.load(json.loads(f.read()))
        return result

    def __str__(self):
        pprint.pformat(self)
