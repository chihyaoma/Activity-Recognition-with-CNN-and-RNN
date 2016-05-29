# setup code

# notes:
# hello: module name in Python corresponding to "color_flow.cpp"

# Contact:
# Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

# Last update: 05/29/2016

#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="PackageName",
      ext_modules=[
          Extension("ColorFlow", ["color_flow.cpp", "colorcode.cpp", "flowIO.cpp",
                                  "Image.cpp", "ImageIO.cpp", "RefCntMem.cpp"],
                    include_dirs=['./headers'],
                    libraries=["boost_python"])
      ])
