import os
import sys

import numpy as np

while True:
    p_id = np.random.randint(1, 100)
    os.system(f"python dataGenerator.py --headless --pid {p_id}")
    # os.system("xvfb-run -a python dataGenerator.py --headless --pid {}".format(sys.argv[1]))