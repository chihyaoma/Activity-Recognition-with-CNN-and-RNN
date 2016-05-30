# Load C++ function to Python
This page teach you how to use [Boost.Python](http://www.boost.org/doc/libs/1_61_0/libs/python/doc/html/index.html) to load C++ functions to Python. We use the Middlebury color encoding as the example.

---
## Requirement
#### Installation
Install the Boost.Python by the following command:
```
$ sudo apt-get install libboost-python-dev
$ sudo apt-get install python-dev
```

#### Code
###### cpp files
In the end of your code, you need to add codes similar as follows:
```
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(ColorFlow)
{
    def("flow2color", flow2color); // depend on your C++ functions
}
```

###### setup.py
You also need an additional Python code:
```
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
```

---
## Usage
You can build your module by the following command:
```
$ python setup.py build
```
If you succeed, it will generate a folder *build*, and you can find the generated module (\*.so) inside. Place this file in an appropriate path and then you can import it in your Python code.

---
#### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) at <cmhungsteve@gatech.edu>

Last updated: 05/30/2016
