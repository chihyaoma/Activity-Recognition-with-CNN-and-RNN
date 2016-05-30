# Test the C++ function in Python

# notes:
# "say_hello": function name defined in "color_coding.cpp"

# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

# Last update: 05/29/2016

import sys
sys.path.insert(1, './build/lib.linux-x86_64-2.7/')

import ColorFlow
ColorFlow.say_hello("World")
