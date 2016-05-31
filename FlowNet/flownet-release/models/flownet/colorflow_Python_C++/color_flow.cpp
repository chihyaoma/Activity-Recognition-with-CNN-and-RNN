// Visualize the flowmap with the standard Middlebury color encoding
// input: one flowmap (*.flo)
// output: image (*.ppm)

// Contact:
// Min-Hung (Steve) Chen at <cmhungsteve@gatech.edu>

// Last update: 05/30/2016

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

int verbose = 1;

void MotionToColor(CFloatImage motim, CByteImage &colim, float maxmotion)
{
    CShape sh = motim.Shape();
    int width = sh.width, height = sh.height;
    colim.ReAllocate(CShape(width, height, 3));
    int x, y;
    // determine motion range:
    float maxx = -999, maxy = -999;
    float minx =  999, miny =  999;
    float maxrad = -1;
    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    if (unknown_flow(fx, fy))
		continue;
	    maxx = __max(maxx, fx);
	    maxy = __max(maxy, fy);
	    minx = __min(minx, fx);
	    miny = __min(miny, fy);
	    float rad = sqrt(fx * fx + fy * fy);
	    maxrad = __max(maxrad, rad);
	}
    }
    printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
	   maxrad, minx, maxx, miny, maxy);


    if (maxmotion > 0) // i.e., specified on commandline
	maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
	maxrad = 1;

    if (verbose)
	fprintf(stderr, "normalizing by %g\n", maxrad);

    for (y = 0; y < height; y++) {
	for (x = 0; x < width; x++) {
	    float fx = motim.Pixel(x, y, 0);
	    float fy = motim.Pixel(x, y, 1);
	    uchar *pix = &colim.Pixel(x, y, 0);
	    if (unknown_flow(fx, fy)) {
		pix[0] = pix[1] = pix[2] = 0;
	    } else {
		computeColor(fx/maxrad, fy/maxrad, pix);
	    }
	}
    }
}

void flow2color(const char *outname, const char *flowname)
{
	float maxmotion = -1;
	CFloatImage im;
	ReadFlowFile(im, flowname);
	CByteImage outim;
	CShape sh = im.Shape();
	sh.nBands = 3;
	outim.ReAllocate(sh);
	outim.ClearPixels();
	MotionToColor(im, outim, maxmotion);
	WriteImageVerb(outim, outname, verbose);
}

//====== Python <---> C++ ======//

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(ColorFlow)
{
    def("say_hello", say_hello);
    def("flow2color", flow2color);
}