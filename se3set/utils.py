import yaml


def save_config(config, path):
    with open(path, 'w') as f:
        yaml.dump(config, f)