///////////////////////////////////////////////////////////////////////////
//
// NAME
//  ImageIO.cpp -- image file input/output
//
// DESCRIPTION
//  Read/write image files, potentially using an interface to an
//  external package.
//
//  File formats currently supported:
//  - a subset of Targa
//  - PGM (1 band) and PPM (4 band)
//  - PMF (multiband float) - homegrown, non-standard
//        (PFM already taken by postscript font maps)
//  - PNG (requires ImageIOpng.cpp, and pnglib and zlib packages)
//
// SEE ALSO
//  ImageIO.h            longer description
//  ImageIOpng.cpp       png reader/writer
//
// Copyright © Richard Szeliski and Daniel Scharstein, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include "Image.h"
#include "Error.h"
#include "ImageIO.h"
#include <vector>

// Comment out next line if you don't have the PNG library
// #define HAVE_PNG_LIB

#ifdef HAVE_PNG_LIB
// implemented in ImageIOpng.cpp
void ReadFilePNG(CByteImage& img, const char* filename);
void WriteFilePNG(CByteImage img, const char* filename);
#endif



// Comment out next line if not using jpeg library
//#define HAVE_JPEG_READER

#ifdef HAVE_JPEG_READER
// implemented in ReadJpg.cpp
void ReadFileJPG(CByteImage& img, const char* filename);
#endif

//
//  Truevision Targa (TGA):  support 24 bit RGB and 32-bit RGBA files
//

typedef unsigned char uchar;

struct CTargaHead
{
    uchar idLength;     // number of chars in identification field
    uchar colorMapType;	// color map type
    uchar imageType;	// image type code
    uchar cMapOrigin[2];// color map origin
    uchar cMapLength[2];// color map length
    uchar cMapBits;     // color map entry size
    short x0;			// x-origin of image
    short y0;			// y-origin of image
    short width;		// width of image
    short height;		// height of image
    uchar pixelSize;    // image pixel size
    uchar descriptor;   // image descriptor byte
};

// Image data type codes
const int TargaRawColormap	= 1;
const int TargaRawRGB		= 2;
const int TargaRawBW		= 3;
const int TargaRunColormap	= 9;
const int TargaRunRGB		= 10;
const int TargaRunBW		= 11;

// Descriptor fields
const int TargaAttrBits     = 15;
const int TargaScreenOrigin = (1<<5);
const int TargaCMapSize		= 256;
const int TargaCMapBands    = 3;
/*
const int TargaInterleaveShift = 6;
const int TargaNON_INTERLEAVE	0
const int TargaTWO_INTERLEAVE	1
const int TargaFOUR_INTERLEAVE	2
const int PERMUTE_BANDS		1
*/

class CTargaRLC
{
    // Helper class to decode run-length-coded image data
public:
    CTargaRLC(bool RLC) : m_count(0), m_RLC(RLC) {}
    uchar* getBytes(int nBytes, FILE *stream);
private:
    int m_count;        // remaining count in current run
    bool m_RLC;         // is stream run-length coded?
    bool m_isRun;       // is current stream of pixels a run?
    uchar m_buffer[4];  // internal buffer
};

inline uchar* CTargaRLC::getBytes(int nBytes, FILE *stream)
{
    // Get one pixel, which consists of nBytes
    if (nBytes > 4)
        throw CError("ReadFileTGA: only support pixels up to 4 bytes long");

    if (! m_RLC)
    {
    	if ((int)fread(m_buffer, sizeof(uchar), nBytes, stream) != nBytes)
    	    throw CError("ReadFileTGA: file is too short");
    }
    else
    {
        if (m_count == 0)
        {
            // Read in the next run count
            m_count = fgetc(stream);
            m_isRun = (m_count & 0x80) != 0;
            m_count = (m_count & 0x7f)  + 1;
            if (m_isRun)  // read the pixels for this run
    	        if ((int)fread(m_buffer, sizeof(uchar), nBytes, stream) != nBytes)
    	            throw CError("ReadFileTGA: file is too short");
        }
        if (! m_isRun)
        {
    	    if ((int)fread(m_buffer, sizeof(uchar), nBytes, stream) != nBytes)
    	        throw CError("ReadFileTGA: file is too short");
        }
        m_count -= 1;
    }
    return m_buffer;
}

void ReadFileTGA(CByteImage& img, const char* filename)
{
    // Open the file and read the header
    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
        throw CError("ReadFileTGA: could not open %s", filename);
    CTargaHead h;
    if ((int)fread(&h, sizeof(CTargaHead), 1, stream) != 1)
	    throw CError("ReadFileTGA(%s): file is too short", filename);

    // Throw away the image descriptor
    if (h.idLength > 0)
    {
        char* tmp = new char[h.idLength];
        int nread = (int)fread(tmp, sizeof(uchar), h.idLength, stream);
        delete tmp;   // throw away this data
        if (nread != h.idLength)
	        throw CError("ReadFileTGA(%s): file is too short", filename);
    }
    //bool isRun = (h.imageType & 8) != 0;
    bool reverseRows = (h.descriptor & TargaScreenOrigin) != 0;
    int fileBytes = (h.pixelSize + 7) / 8;

    // Read the colormap
    uchar colormap[TargaCMapSize][TargaCMapBands];
    int cMapSize = 0;
    bool grayRamp = false;
    if (h.colorMapType == 1)
    {
        cMapSize = (h.cMapLength[1] << 8) + h.cMapLength[0];
        if (h.cMapBits != 24)
            throw CError("ReadFileTGA(%s): only 24-bit colormap currently supported", filename);
	    int l = fileBytes * cMapSize;
        if (l > TargaCMapSize * TargaCMapBands)
	        throw CError("ReadFileTGA(%s): colormap is too large", filename);
	    if ((int)fread(colormap, sizeof(uchar), l, stream) != l)
	        throw CError("ReadFileTGA(%s): could not read the colormap", filename);

        // Check if it's just a standard gray ramp
	int i;
        for (i = 0; i < cMapSize; i++) {
            for (int j = 0; j < TargaCMapBands; j++)
                if (colormap[i][j] != i)
                    break;
        }
        grayRamp = (i == cMapSize);    // didn't break out too soon
    }
    bool isGray = 
        h.imageType == TargaRawBW || h.imageType == TargaRunBW ||
        (grayRamp &&
	 (h.imageType == TargaRawColormap || h.imageType == TargaRunColormap));
    bool isRaw = h.imageType == TargaRawBW || h.imageType == TargaRawRGB ||
        (h.imageType == TargaRawRGB && isGray);

    // Determine the image shape
    CShape sh(h.width, h.height, (isGray) ? 1 : 4);
    
    // Allocate the image if necessary
    img.ReAllocate(sh, false);

    // Construct a run-length code reader
    CTargaRLC rlc(! isRaw);

    // Read in the rows
    for (int y = 0; y < sh.height; y++)
    {
        int yr = reverseRows ? sh.height-1-y : y;
        uchar* ptr = (uchar *) img.PixelAddress(0, yr, 0);
        if (fileBytes == sh.nBands && isRaw)
        {
            // Special case for raw image, same as destination
            int n = sh.width*sh.nBands;
    	    if ((int)fread(ptr, sizeof(uchar), n, stream) != n)
    	        throw CError("ReadFileTGA(%s): file is too short", filename);
        }
        else
        {
            // Read one pixel at a time
            for (int x = 0; x < sh.width; x++, ptr += sh.nBands)
            {
                uchar* buf = rlc.getBytes(fileBytes, stream);
                if (fileBytes == 1 && sh.nBands == 1)
                {
                    ptr[0] = buf[0];
                }
                else if (fileBytes == 1 && sh.nBands == 4)
                {
                    for (int i = 0; i < 3; i++)
                        ptr[i] = (isGray) ? buf[0] : colormap[buf[0]][i];
                    ptr[3] = 255;   // full alpha;
                }
                else if ((fileBytes == 3 || fileBytes == 4) && sh.nBands == 4)
                {
                    int i;
                    for (i = 0; i < fileBytes; i++)
                        ptr[i] = buf[i];
                    if (i == 3) // missing alpha channel
                        ptr[3] = 255;   // full alpha;
                }
                else
            	    throw CError("ReadFileTGA(%s): unhandled pixel depth or # of bands", filename);
            }
        }
    }

    if (fclose(stream))
        throw CError("ReadFileTGA(%s): error closing file", filename);
}

void WriteFileTGA(CImage img, const char* filename)
{
    // Only 1, 3, or 4 bands supported
    CShape sh = img.Shape();
    int nBands = sh.nBands;
    if (nBands != 1 && nBands != 3 && nBands != 4)
        throw CError("WriteFileTGA(%s): can only write 1, 3, or 4 bands", filename);

    // Only unsigned_8 supported directly
#if 0   // broken for now
    if (img.PixType() != unsigned_8)
    {
        CImage u8img(sh, unsigned_8);
        TypeConvert(img, u8img);
        img = u8img;
    }
#endif

    // Fill in the header structure
    CTargaHead h;
    memset(&h, 0, sizeof(h));
    h.imageType = (nBands == 1) ? TargaRawBW : TargaRawRGB;
        // TODO:  is TargaRawBW the right thing, or only binary?
    h.width     = sh.width;
    h.height    = sh.height;
    h.pixelSize = 8 * nBands;
    bool reverseRows = false;   // TODO: when is this true?

    // Open the file and write the header
    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
        throw CError("WriteFileTGA: could not open %s", filename);
    if (fwrite(&h, sizeof(CTargaHead), 1, stream) != 1)
	    throw CError("WriteFileTGA(%s): file is too short", filename);

    // Write out the rows
    for (int y = 0; y < sh.height; y++)
    {
        int yr = reverseRows ? sh.height-1-y : y;
        char* ptr = (char *) img.PixelAddress(0, yr, 0);
        int n = sh.width*sh.nBands;
    	if ((int)fwrite(ptr, sizeof(uchar), n, stream) != n)
    	    throw CError("WriteFileTGA(%s): file is too short", filename);
    }

    if (fclose(stream))
        throw CError("WriteFileTGA(%s): error closing file", filename);
}

//
// Portable Graymaps: support PGM, PPM, and PMF images
//

void skip_comment(FILE *fp)
{
    // skip comment lines in the headers of pnm files

    char c;
    while ((c=getc(fp)) == '#')
        while (getc(fp) != '\n');
    ungetc(c, fp);
}

void skip_space(FILE *fp)
{
    // skip white space in the headers or pnm files

    char c;
    do {
        c = getc(fp);
    } while (c == '\n' || c == ' ' || c == '\t' || c == '\r');
    ungetc(c, fp);
}

void read_header(FILE *fp, const char *imtype, char c1, char c2, 
                 int *width, int *height, int *nbands, int thirdArg)
{
    // read the header of a pnmfile and initialize width and height

    char c;
  
	if (getc(fp) != c1 || getc(fp) != c2)
		throw CError("ReadFilePGM: wrong magic code for %s file", imtype);
	skip_space(fp);
	skip_comment(fp);
	skip_space(fp);
	fscanf(fp, "%d", width);
	skip_space(fp);
	fscanf(fp, "%d", height);
	if (thirdArg) {
		skip_space(fp);
		fscanf(fp, "%d", nbands);
	}
    // skip SINGLE newline character after reading image height (or third arg)
	c = getc(fp);
    if (c == '\r')      // <cr> in some files before newline
        c = getc(fp);
    if (c != '\n') {
        if (c == ' ' || c == '\t' || c == '\r')
            throw CError("newline expected in file after image height");
        else
            throw CError("whitespace expected in file after image height");
  }
}


void ReadFilePGM(CByteImage& img, const char* filename)
{
    // Open the file and read the header
    FILE *stream = fopen(filename, "rb");
    if (stream == 0)
        throw CError("ReadFilePGM: could not open %s", filename);

	int width, height, nBands;
	const char *dot = strrchr(filename, '.');
	int isGray = 0, isFloat = 0;

    if (strcmp(dot, ".pgm") == 0) {
		read_header(stream, "PGM", 'P', '5', &width, &height, &nBands, 1);
		isGray = 1;
	}
    else if (strcmp(dot, ".ppm") == 0) {
		read_header(stream, "PGM", 'P', '6', &width, &height, &nBands, 1);
		isGray = 0;
	}
    else if (strcmp(dot, ".pmf") == 0) {
		read_header(stream, "PMF", 'P', '9', &width, &height, &nBands, 1);
		isGray = 0;
        isFloat = 1;
	}


    // Determine the image shape
    CShape sh(width, height, (isGray) ? 1 : (isFloat) ? nBands : 4);
    
    // Allocate the image if necessary
    if (isFloat)
        ((CFloatImage *) &img)->ReAllocate(sh);
    else
        img.ReAllocate(sh);
 
    if (isGray || isFloat) { // read PGM or PMF
		
		// read the rows
        int n = isFloat ? width * nBands * sizeof(float) : sh.width;
		for (int y = 0; y<sh.height; y++) {
			uchar* ptr = (uchar *) img.PixelAddress(0, y, 0);
    	    if ((int)fread(ptr, sizeof(uchar), n, stream) != n)
    	        throw CError("ReadFilePGM(%s): file is too short", filename);
		}
	}
    else { // read PPM

		// read the rows
        int n = sh.width*3;
		std::vector<uchar> rowBuf;
		rowBuf.resize(n);
		for (int y = 0; y<sh.height; y++) {
	   	    if ((int)fread(&rowBuf[0], sizeof(uchar), n, stream) != n)
    	        throw CError("ReadFilePGM(%s): file is too short", filename);

			uchar* ptr = (uchar *) img.PixelAddress(0, y, 0);
			int x = 0;
			while (x < n) {
				ptr[2] = rowBuf[x++];
				ptr[1] = rowBuf[x++];
				ptr[0] = rowBuf[x++];
				ptr[3] = 255; // full alpha
				ptr += 4;
			}
		}
	}

    if (fclose(stream))
        throw CError("ReadFilePGM(%s): error closing file", filename);
}



void WriteFilePGM(CByteImage img, const char* filename)
{
    // Write a PGM, PPM, or PMF file
    CShape sh = img.Shape();
    int nBands = sh.nBands;
    int isFloat = img.PixType() == typeid(float);
  
	// Determine the file extension
    const char *dot = strrchr(filename, '.');
	if (strcmp(dot, ".pgm") == 0 && nBands != 1)
		throw CError("WriteFilePGM(%s): can only write 1-band image as pgm", filename);
	
	if (strcmp(dot, ".ppm") == 0 && nBands != 3 && nBands != 4)
		throw CError("WriteFilePGM(%s): can only write 3 or 4-band image as ppm", filename);

	if (strcmp(dot, ".pmf") == 0 && ! isFloat)
		throw CError("WriteFilePMF(%s): can only write floating point image as pmf", filename);
    // Open the file
    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
        throw CError("WriteFilePGM: could not open %s", filename);

    if (nBands == 1 || isFloat) { // write PGM or PMF
		
		// write the header
        fprintf(stream, "P%d\n%d %d\n%d\n", isFloat ? 9 : 5, sh.width, sh.height,
                isFloat ? sh.nBands : 255);

		// write the rows
        int n = isFloat ? sh.width * sh.nBands * sizeof(float) : sh.width;
		for (int y = 0; y<sh.height; y++) {
			char* ptr = (char *) img.PixelAddress(0, y, 0);
    		if ((int)fwrite(ptr, sizeof(uchar), n, stream) != n)
    			throw CError("WriteFilePGM(%s): file is too short", filename);
		}
	}
    /*
    else if (nBands == 3) { // write PPM
		
		// write the header
		fprintf(stream, "P6\n%d %d\n%d\n", sh.width, sh.height, 255);

		// write the rows
        int n = sh.width*3;
		for (int y = 0; y<sh.height; y++) {
			char* ptr = (char *) img.PixelAddress(0, y, 0);
    		if ((int)fwrite(ptr, sizeof(uchar), n, stream) != n)  // ??? TODO: test this
                // ??? not sure this will work - RGB may be in wrong order
    			throw CError("WriteFilePGM(%s): file is too short", filename);
		}
	}
    */
    else if (nBands == 3 || nBands == 4) { // write PPM, ignoring alpha
		
		// write the header
		fprintf(stream, "P6\n%d %d\n%d\n", sh.width, sh.height, 255);

		// write the rows
        int n = sh.width*3;
		std::vector<uchar> rowBuf;
		rowBuf.resize(n);
		for (int y = 0; y<sh.height; y++) {
			uchar* ptr = (uchar *) img.PixelAddress(0, y, 0);
			int x = 0;
			while (x < n) {
				rowBuf[x++] = ptr[2];
				rowBuf[x++] = ptr[1];
				rowBuf[x++] = ptr[0];
				ptr += nBands;
			}

    		if ((int)fwrite(&rowBuf[0], sizeof(uchar), n, stream) != n)
    			throw CError("WriteFilePGM(%s): file is too short", filename);
		}
	}
	else
        throw CError("WriteFilePGM(%s): unhandled # of bands %d", filename, nBands);

	// close file

    if (fclose(stream))
        throw CError("WriteFilePGM(%s): error closing file", filename);
}


//
// main dispatch functions
//

void ReadImage (CImage& img, const char* filename)
{
    if (filename == NULL)
	throw CError("ReadImage: empty filename");

    // Determine the file extension
    const char *dot = strrchr(filename, '.');
    if (dot == NULL)
	throw CError("ReadImage: extension required in filename '%s'", filename);

    if (strcmp(dot, ".TGA") == 0 || strcmp(dot, ".tga") == 0)
    {
        if ((&img.PixType()) == 0)
            img.ReAllocate(CShape(), typeid(uchar), sizeof(uchar), true);
        if (img.PixType() == typeid(uchar))
            ReadFileTGA(*(CByteImage *) &img, filename);
        else
           throw CError("ReadImage(%s): can only read CByteImage in TGA format", filename);
    }
    else if (strcmp(dot, ".pgm") == 0 || strcmp(dot, ".ppm") == 0 ||
             strcmp(dot, ".pmf") == 0)
    {
        if ((&img.PixType()) == 0)
        {
            if (strcmp(dot, ".pmf") == 0)
                img.ReAllocate(CShape(), typeid(float), sizeof(float), true);
            else
                img.ReAllocate(CShape(), typeid(uchar), sizeof(uchar), true);
        }
        if (img.PixType() == typeid(uchar) ||
            img.PixType() == typeid(float))
            ReadFilePGM(*(CByteImage *) &img, filename);
        else
           throw CError("ReadImage(%s): wrong image type for PGM/PPM/PMF", filename);
    }
#ifdef HAVE_PNG_LIB
    else if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0)
    {
        if ((&img.PixType()) == 0)
            img.ReAllocate(CShape(), typeid(uchar), sizeof(uchar), true);
        if (img.PixType() == typeid(uchar))
            ReadFilePNG(*(CByteImage *) &img, filename);
        else
           throw CError("ReadImage(%s): can only read CByteImage in PNG format", filename);
    }
#endif
#ifdef HAVE_JPEG_READER
    else if (strcmp(dot, ".JPG") == 0 || strcmp(dot, ".jpg") == 0)
    {
        if ((&img.PixType()) == 0)
            img.ReAllocate(CShape(), typeid(uchar), sizeof(uchar), true);
        if (img.PixType() == typeid(uchar))
            ReadFileJPG(*(CByteImage *) &img, filename);
        else
           throw CError("ReadImage(%s): can only read CByteImage in JPG format", filename);
    }
#endif
    else
        throw CError("ReadImage(%s): file type not supported", filename);
}



void WriteImage(CImage& img, const char* filename)
{
    if (filename == NULL)
	throw CError("WriteImage: empty filename");

    // Determine the file extension
    const char *dot = strrchr(filename, '.');
    if (dot == NULL)
	throw CError("WriteImage: extension required in filename '%s'", filename);

    if (strcmp(dot, ".TGA") == 0 || strcmp(dot, ".tga") == 0)
    {
        if (img.PixType() == typeid(uchar))
            WriteFileTGA(*(CByteImage *) &img, filename);
        else
           throw CError("WriteImage(%s): can only write CByteImage in TGA format", filename);
    }
    else if (strcmp(dot, ".pgm") == 0 || strcmp(dot, ".ppm") == 0 ||
             strcmp(dot, ".pmf") == 0)
    {
        if (img.PixType() == typeid(uchar) ||
            img.PixType() == typeid(float))
            WriteFilePGM(*(CByteImage *) &img, filename);
        else
           throw CError("WriteImage(%s): wrong image type for PGM/PPM/PMF", filename);
    }
#ifdef HAVE_PNG_LIB
    else if (strcmp(dot, ".PNG") == 0 || strcmp(dot, ".png") == 0)
    {
        if (img.PixType() == typeid(uchar))
            WriteFilePNG(*(CByteImage *) &img, filename);
        else
           throw CError("WriteImage(%s): can only write CByteImage in PNG format", filename);
    }
#endif
    else
        throw CError("WriteImage(%s): file type not supported", filename);
}

// read an image and perhaps tell the user you're doing so
void ReadImageVerb(CImage& img, const char* filename, int verbose) {
	if (verbose)
		fprintf(stderr, "Reading image %s\n", filename);
	ReadImage(img, filename);
}

// write out an image and perhaps tell the user you're doing so
void WriteImageVerb(CImage& img, const char* filename, int verbose) {
	if (verbose)
		fprintf(stderr, "Writing image %s\n", filename);
	WriteImage(img, filename);
}
