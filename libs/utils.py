import yaml
from munch import munchify
import random
from string import ascii_lowercase
import os
from glob import glob
import re
import torch
import datetime


# output current timestamp with message to console
def timestamp():
    st = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S |")
    return st


# output reported arguments to log file
def report(*args):
    string = timestamp()
    for arg in args:
        if isinstance(arg, str):
            string += " " + arg
        else:
            string += " " + str(arg)
    print(string)
    # log = open("run.log", "a+")
    # log.write(string + "\n")
    # log.close()
    return True


def load_yaml(path="config.yaml"):
    if not path.endswith(".yaml"):
        path += ".yaml"
    with open(path) as stream:
        config = yaml.safe_load(stream)
    return munchify(config)


def get_random_hash():
    return "".join(random.choice(ascii_lowercase) for i in range(10))


def checkpoint(id=None, data=None, path="./checkpoints", cleanup=True):
    if not os.path.isdir(path):
        os.mkdir(path)
    path = os.path.join(path, id)
    if not os.path.isdir(path):
        os.mkdir(path)
    removables = glob(os.path.join(path, f"{id}*"))
    if len(removables) > 0:
        latest = os.path.basename(max(removables, key=os.path.getctime))
        try:
            counter = int(re.search(r"(\d{3})\.dict$", latest).groups()[0])
        except Exception:
            counter = 0
    else:
        counter = 0
    if id is None or data is None:
        print("Missing checkpoint parameters, skipping")
        return False
    if not os.path.isdir(path):
        os.mkdir(path)
    chkptfname = os.path.join(path, f"{id}{(counter+1):03}.dict")
    torch.save(data, chkptfname)
    if not os.path.isfile(chkptfname):
        raise IOError(f"Checkpoint {chkptfname} was not saved")
    else:
        report(f"Checkpoint {chkptfname} saved")
    if cleanup:
        for rm in removables:
            os.remove(rm)
    if not os.path.isfile(chkptfname):
        raise IOError(f"Checkpoint {chkptfname} was deleted by cleanup")
