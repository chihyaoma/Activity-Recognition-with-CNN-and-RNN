//Generate the optical flow video
//single video
//
//author: Min-Hung Chen
//contact: cmhungsteve@gatech.edu
//Last updated: 06/30/2016

#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
//#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>

//#include "tick_meter.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void showFlow(string name, const GpuMat& d_flow, Mat& out)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    drawOpticalFlow(flowx, flowy, out, 10);

    //imshow(name, out);


}

int main(int argc, const char* argv[])
{
    string nameVideo;
    string type("CPU"); // 1: CPU; 2. GPU
    int numStep = 1; // step for calculating the optical flow

    // check the arguments
    if (argc != 2)
    {
    	cerr << "Usage : " << argv[0] << " <video name> " << endl;
    	nameVideo = "v_HorseRiding_g01_c01";
    }
    else
    {
    	nameVideo = argv[1];
    }

    string nameFlow(nameVideo + "_flow");

    //namedWindow("GPU", cv::WINDOW_OPENGL);
    //setGlDevice();

    //cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    if(!type.compare("CPU"))
    {
    	VideoCapture reader(nameVideo + ".avi");

    	// video information
    	double FPS = reader.get(CV_CAP_PROP_FPS)/numStep; // frame rate

    	// read the first frame
    	Mat framePrvs;
    	reader.read(framePrvs);

    	// video writer initialization
    	VideoWriter writer(nameFlow + ".avi", CV_FOURCC('X', 'V', 'I', 'D'), FPS, framePrvs.size());
    	if ( !writer.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
    	{
    		cout << "ERROR: Failed to write the video" << endl;
    		return -1;
    	}

    	GpuMat g_flow(framePrvs.size(), CV_32FC2);

    	int countFrame = 0;
    	int countStep = 0;
//    	double timeAcc = 0; // accumulate the computation time

    	const int64 start = getTickCount();

    	// read the next frame & calculate the optical flow
    	for (;;)
    	{
//    	    const int64 startFrame = getTickCount();

    	    Mat frameNext;
   	        if (!reader.read(frameNext))
   	        {
   	        	cout << "Cannot read the frame" << endl;
   	        	break;
   	        }
   	        countFrame++;
   	        countStep++;

   	        if (countStep == numStep)
   	        {
   	        	Mat framePrvsGray;
   	        	cvtColor(framePrvs, framePrvsGray, CV_BGR2GRAY); // convert to grayscale

   	        	// convert to GPU & float --> compute optical flow
   	        	Mat frameNextGray;
   	        	cvtColor(frameNext, frameNextGray, CV_BGR2GRAY); // convert to grayscale
   	        	GpuMat g_framePrvs(framePrvsGray); // convert to GPU
   	        	GpuMat g_frameNext(frameNextGray);

   	        	GpuMat framePrvs_f;
   	        	GpuMat frameNext_f;

   	        	g_framePrvs.convertTo(framePrvs_f, CV_32F, 1.0 / 255.0);
   	        	g_frameNext.convertTo(frameNext_f, CV_32F, 1.0 / 255.0);

   	        	brox->calc(framePrvs_f, frameNext_f, g_flow);

//   	        	const double timeSecFrame = (getTickCount() - startFrame) / getTickFrequency();
//   	        	//    	    cout << "Brox : " << timeSecFrame << " sec" << endl;
//   	        	timeAcc = timeAcc + timeSecFrame;

   	        	// Color coding (Middlebury)
   	        	Mat flow_out;
   	        	showFlow(nameFlow + " (Brox)", g_flow, flow_out);

   	        	// show the frames & result flow map
   	        	Mat imgDisplay;
   	        	hconcat(framePrvs, frameNext, imgDisplay);
   	        	hconcat(imgDisplay, flow_out, imgDisplay);

   	        	imshow(nameFlow + " (Brox)", imgDisplay);

   	        	if (waitKey(3) > 0)
   	        		break;

   	        	// save video
   	        	writer.write(flow_out);

   	        	framePrvs = frameNext;
   	        	countStep = 0;
   	        }
    	}

//    	cout << "Brox (per frame): " << timeAcc/countFrame << " sec" << endl;

    	const double timeSecFrame = (getTickCount() - start) / getTickFrequency();
    	cout << "Computation time for " + nameFlow + " (Brox): " << timeSecFrame << " sec" << endl;

    }

    else if(!type.compare("GPU"))
    {
    	Ptr<cv::cudacodec::VideoReader> g_reader = cv::cudacodec::createVideoReader(nameVideo);
    	// read the first frame
    	GpuMat g_framePrvs;
    	g_reader->nextFrame(g_framePrvs);

    	GpuMat g_flow(g_framePrvs.size(), CV_32FC2);

        // read the next frame & calculate the optical flow
        for (;;)
        {
//        	const int64 start = getTickCount();

        	cvtColor(g_framePrvs, g_framePrvs, CV_BGR2GRAY); // convert to grayscale

        	GpuMat g_frameNext;
            if (!g_reader->nextFrame(g_frameNext))
            {
            	cout << "Cannot read the frame" << endl;
            	break;
            }

       	    GpuMat framePrvs_f;
       	    GpuMat frameNext_f;

       	    g_framePrvs.convertTo(framePrvs_f, CV_32F, 1.0 / 255.0);
       	    g_frameNext.convertTo(frameNext_f, CV_32F, 1.0 / 255.0);

       	    brox->calc(framePrvs_f, frameNext_f, g_flow);

//            const double timeSec = (getTickCount() - start) / getTickFrequency();
//            cout << "Brox : " << timeSec << " sec" << endl;

       	    Mat flow_out;
       	    showFlow("Brox", g_flow, flow_out);
        	if (cv::waitKey(3) > 0)
        		break;

        	g_framePrvs = g_frameNext;

        }
    }

    return 0;
}
