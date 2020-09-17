import os
import sys

import numpy as np

while True:
    os.system("python dataGenerator.py --headless --pid {}".format(sys.argv[1]))
    # os.system("xvfb-run -a python dataGenerator.py --headless --pid {}".format(sys.argv[1]))