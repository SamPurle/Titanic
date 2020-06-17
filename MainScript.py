"""

Titanic:
    
    A Machine Learning model to predict whether passengers survived the Titanic.
    
"""

# Import libraries

from os import system
import time
from subprocess import call

# Specify parameters

global NULL_THRESH
NULL_THRESH = 0.25  # Specify the threshold for dropping null columns

global FANCY_THRESH
FANCY_THRESH = 10 # Specify the maximum number of people for a title to be considered fancy

"""

Titanic - Cleaning:
    
    A script to clean the Raw data intto a usable format
    
"""

st = time.time()
FILENAME = 'Cleaning'
system('python {}.py'.format(FILENAME))
print('{} complete. The completion time was {:.1f} seconds'.format(
    FILENAME, time.time() - st))