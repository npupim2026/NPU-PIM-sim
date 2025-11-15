from __future__ import absolute_import, division, print_function, unicode_literals


class AttrDict(dict):
    # Simple Attribute dictionary for the configuration
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

def from_dict_to_attrdict(target):
    if isinstance(target, dict):
        ret = AttrDict()
        for key in target:
            ret[key] = from_dict_to_attrdict(target[key])
        return ret
    else:
        return target

def from_attrdict_to_dict(target):
    if isinstance(target, dict):
        ret = {}
        for key in target:
            ret[key] = from_attrdict_to_dict(target[key])
        return ret
    else:
        return target
