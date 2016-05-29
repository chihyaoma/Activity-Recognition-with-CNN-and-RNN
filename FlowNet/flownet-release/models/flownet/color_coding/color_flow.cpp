// Visualize the flowmap with the standard Middlebury color encoding
// input: one flowmap (*.flo)
// output: image (*.ppm)

// Contact:
// Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

// Last update: 05/29/2016

#include <iostream>
#include <cstdio>
#include <cmath>

#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"

using namespace std;

void say_hello(const char* name) {
    cout << "Hello " <<  name << "!\n";
}

//====== Python <---> C++ ======//

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(ColorFlow)
{
    def("say_hello", say_hello);
}