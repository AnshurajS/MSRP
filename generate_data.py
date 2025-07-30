# ~/MARGE/generate_data.py

import os, numpy as np
from MARGE.trial_mcmc import w_trans, w_emit, e_trans, e_emit
from .prt_forward import forward_model
import configparser

def generate_data(cfile):
    """
    This will be called by MARGE when `datagen=True`.
    `conf` is your ConfigParser for the [MARGE] section.
    """
    # read in your own section of the cfg
    parser = configparser.ConfigParser()
    parser.read(cfile)

    try:
        conf = parser['MARGE']
        which = conf.get("which", "both")
    except KeyError:
        print("KeyError: MARGE section not found. Available sections:", list(parser.sections()))
        print("Trying alternative section names...")
        # Try alternative approaches
        if 'MARGE' in parser.sections():
            conf = parser['MARGE']
        elif len(parser.sections()) > 0:
            # Use the first section if MARGE not found
            section_name = parser.sections()[0]
            print(f"Using section: {section_name}")
            conf = parser[section_name]
        else:
            raise ValueError("No sections found in configuration file")

    # read out the data directory roots from the config:
    datadir = os.path.abspath(conf["datadir"])      # e.g. "data"
    train_dir = os.path.join(datadir, "train")
    valid_dir = os.path.join(datadir, "valid")
    test_dir  = os.path.join(datadir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    # Here: define how many samples you want in each split
    N_train = int(conf.get("n_train", 10000))
    N_valid = int(conf.get("n_valid",   2000))
    N_test  = int(conf.get("n_test",    2000))

    # A helper to draw random ? in your prior and dump x,y into .npz
    def make_split(N, out_dir):
        for i in range(N):
            # draw ? uniformly in your prior bounds:
            theta = np.random.uniform(-12, -1, size=4)   # [logX_H2O, logX_CH4, logX_CO, logX_CO2]
            wl_t, depth_t, wl_e, flux_e = forward_model(theta)

            # construct your x,y arrays exactly matching your MCMC and MARGE expectations:
            y_t = np.interp(w_trans, wl_t, depth_t)[None, :]
            y_e = np.interp(w_emit, wl_e, flux_e)[None, :]

            # save:
            if which in ("trans","both"):
                np.savez(os.path.join(out_dir, f"t_{i:04d}.npz"),
                         x=theta[None,:], y=y_t)
            if which in ("emit","both"):
                np.savez(os.path.join(out_dir, f"e_{i:04d}.npz"),
                         x=theta[None,:], y=y_e)

    print("Generating TRAIN data")
    make_split(N_train, train_dir)
    print("Generating VALID data")
    make_split(N_valid, valid_dir)
    print("Generating TEST data")
    make_split(N_test, test_dir)
    print("Done data generation.")

def process_data(cfile, datadir, preservedat):
    return
