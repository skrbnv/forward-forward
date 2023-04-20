import yaml
from munch import Munch


def load_yaml(path='config.yaml'):
    if not path.endswith('.yaml'):
        path += '.yaml'
    with open(path) as stream:
        config = yaml.safe_load(stream)
    return Munch(config)
