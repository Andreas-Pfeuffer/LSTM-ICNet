

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import numpy as np

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_config(edict, path2dict):

    try:
        with open(path2dict, "w") as file:
            json.dump(json.dumps(edict, cls=NumpyEncoder), file) 
        file.close()
        print ('[save_config] save config to "{}"'.format(path2dict))
    except:
        print ('[save_config] fail to save config to "{}"'.format(path2dict))


def load_config(path2config):

    try:
        with open(path2config, "r") as file:
            config = edict(json.loads(json.load(file)))
        file.close()
    except:
        print ('[load_config] cannot load config "{}"'.format(path2config))
        config = edict()

    return config


