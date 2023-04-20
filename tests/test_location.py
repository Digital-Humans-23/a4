import os, sys, glob
sys.path.append(os.getcwd())

import pytest
import pickle
import numpy as np

def test_random():
    random_log_path = 'results/location_random/quality_log0.pkl'
    with open(random_log_path, 'rb') as f:
        log = pickle.load(f)
    quality = log['location']
    assert 0.9 > quality[-1] > 0.0

def test_search():
    search_log_path = 'results/location_search/quality_log0.pkl'
    with open(search_log_path, 'rb') as f:
        log = pickle.load(f)
    quality = log['location']
    assert quality[-1] > 0.9