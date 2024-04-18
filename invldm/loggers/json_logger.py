import os
import json
import argparse
from copy import deepcopy
from . import BaseLogger
from ..utils.utils import namespace2dict


class JSONLogger(BaseLogger):
    def __init__(self, args):
        super().__init__(args)
        self.hparams = {}
        self.scalars = {}
        
        self.log_file = os.path.join(
            self.args.run.exp_folder,
            "JSON/log.json"
        )
        return None

    def save_logger(self):
        json_dic = {"hyperparameters": self.hparams}
        for key, val in self.scalars.items():
            json_dic.update({key: val})
        
        with open(self.log_file, 'w') as f:
            json.dump(json_dic, f)
        
    def log_scalar(self, tag, val, step, **kwargs):
        try:
            self.scalars[tag].append(val)
            self.scalars[tag+"_step"].append(step)
        except KeyError:
            self.scalars[tag] = [val]
            self.scalars[tag+"_step"] = [step]
        self.save_logger()
        return None

    def log_hparams(self, hparam_dict, metric_dict, **kwargs):
        # hparam_dict.update(metric_dict)
        self.hparams.update(clear_dic_for_json(hparam_dict))
        self.save_logger()
        return None
    

def clear_dic_for_json(dictionary, pop=False):
    clean_dict = deepcopy(dictionary)
    for key, val in dictionary.items():
        if isinstance(val, dict):
            val = clear_dic_for_json(val, pop)
            clean_dict[key] = val

        elif isinstance(val, argparse.Namespace):
            val = namespace2dict(val, flatten=False)
            val = clear_dic_for_json(val, pop)
            clean_dict[key] = val

        if not is_jsonable(val):
            if pop:
                clean_dict.pop(key)
            else:
                clean_dict[key] = str(val)
    return clean_dict


def is_jsonable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError) as e:
        return False