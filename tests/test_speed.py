import os, sys, glob
sys.path.append(os.getcwd())

import pytest
import pickle
import numpy as np

def test_random():
    random_log_path = 'results/speed_random/quality_log0.pkl'
    with open(random_log_path, 'rb') as f:
        log = pickle.load(f)
    quality_mean = np.mean(log['speed'][-10:])
    assert 0.9 > quality_mean > 0.0

def test_search():
    search_log_path = 'results/speed_search/quality_log0.pkl'
    with open(search_log_path, 'rb') as f:
        log = pickle.load(f)
    quality_mean = np.mean(log['speed'][-10:])
    assert quality_mean > 0.9