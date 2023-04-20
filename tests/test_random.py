import os, sys, glob
sys.path.append(os.getcwd())

import pytest

def test():
    motion_path = 'results/random.pkl'
    assert os.path.exists(motion_path)