import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import copy
import os
import random

PAD = 0
EOS = 1
UNK = 2
SOS = 3
T = 1
NUM_BBOX = 36
VISUAL_FEAT = 2048
BIAS = 1000
EMB_DIM = 100



def parse_program(string):
	if '=' in string:
		result, function = string.split('=')
	else:
		function = string
		result = "?"

	func, arguments = function.split('(')
	if len(arguments) == 1:
		return result, func, []
	else:
		arguments = list(map(lambda x:x.strip(), arguments[:-1].split(',')))
		return result, func, arguments

