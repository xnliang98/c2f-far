import os
import json
import pickle as pkl

# IO
def load_jsonline(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [json.loads(line.strip()) for line in lines]


def save_jsonline(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            json.dump(line, f)
            f.writelines('\n')
    
    print(f"Data has been saved into {path}.")
    return data


def load_txt(path, sep=None):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sep) if sep is not None else line.strip() for line in lines]


def save_txt(data, path, sep=None):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            if type(line) == list and sep is not None:
                line = sep.join(line)
            else:
                line = line.strip()
            f.writelines(line + "\n")
    print(f"Data has been saved into {path}.")
    return None

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
    print(f"Data has been saved into {path}.")
    return None

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def make_dirs(path):
    if not os.path.exists(path):
        print(f"{path} does not exist, create it.")
        os.makedirs(path)
    return None
 