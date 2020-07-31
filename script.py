import os
import sys

while True:
    os.system("xvfb-run -a python dataGenerator.py --headless --pid {}".format(sys.argv[1]))