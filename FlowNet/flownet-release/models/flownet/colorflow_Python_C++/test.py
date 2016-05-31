# Test the C++ function in Python

# notes:
# "say_hello": function name defined in "color_coding.cpp"

# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

# Last update: 05/30/2016

import sys
sys.path.insert(1, './build/lib.linux-x86_64-2.7/')

import ColorFlow
ColorFlow.say_hello("World")


nameInput = 'flownetc-pred-0000000.flo'
nameOutput = 'out.ppm'

ColorFlow.flow2color(nameOutput,nameInput)