//Generate optical flow videos for the whole dataset
//Target dataset: UCF-101
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

#include <dirent.h> // for directory/folder processing
#include <sys/stat.h>

#include <boost/algorithm/string.hpp> // for splitting string
//#include <boost/filesystem.hpp> // check files

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace boost;

/* user-defined parameters */
string type("CPU"); // 1: CPU; 2. GPU
int numStep = 1; // step for calculating the optical flow
string methodOF("TVL1"); // Brox; TVL1
int methodCoding = 2; // 1. Middlebury color coding; 2. 2 channels (0 for the 3rd channel)
bool opt_crop = true; // whether to crop the optical flow value
int value_crop = 20;

/* Data Path */
 string dirDatabase("/home/cmhung/Code/Dataset/UCF-101/");
//string dirDatabase("/media/chih-yao/SSD/dataset/UCF-101/");
string inDir(dirDatabase + "RGB/");
string outDir(dirDatabase + "FlowMap-" + methodOF + "-crop20/");

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

    for (int y = 0; y < flowx.rows; ++y) // normalize using Middlebury color coding
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

static void drawOpticalFlow2(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxX = 0;
    float maxY = 0;
    float minX = 10000;
    float minY= 10000;

    for (int y = 0; y < flowx.rows; ++y)
    {
    	for (int x = 0; x < flowx.cols; ++x)
    	{
    		Point2f u(flowx(y, x), flowy(y, x));

    		if (!isFlowCorrect(u))
    			continue;

    		if (opt_crop)
			{
    			u.x = (u.x > value_crop) ? value_crop : u.x;
    			u.y = (u.y > value_crop) ? value_crop : u.y;
    			u.x = (u.x < -value_crop) ? -value_crop : u.x;
    			u.y = (u.y < -value_crop) ? -value_crop : u.y;
			}

    		maxX = max(maxX, u.x);
    		maxY = max(maxY, u.y);
    		minX = min(minX, u.x);
    		minY = min(minY, u.y);
    	}
    }


    for (int y = 0; y < flowx.rows; ++y) // normalization
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
            {
            	float val_x = 255.0 * (u.x - minX) / (maxX - minX);
            	float val_y = 255.0 * (u.y - minY) / (maxY - minY);
            	dst.at<Vec3b>(y, x) = Vec3b(static_cast<uchar>(val_x),static_cast<uchar>(val_y),0);
            }
        }
    }
}

static void showFlow(string name, const GpuMat& d_flow, Mat& out)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    if (methodCoding==1)
    	drawOpticalFlow(flowx, flowy, out, 10);
    else if (methodCoding==2)
    	drawOpticalFlow2(flowx, flowy, out);

    //imshow(name, out);


}

int main(int argc, const char* argv[])
{
	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

	/* Initialize the optical flow algorithm */
	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
	Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create();

	// create the output folder
	if (!opendir(outDir.c_str())) // create the folder if not existed
	{
		mkdir(outDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}

	/* Classes */
	// retrieve all the folders in the database
	DIR           *d;
	struct dirent *dir;
	d = opendir(inDir.c_str());
	vector<string> nameClasses;

	if (d)
	{
		while ((dir = readdir(d)) != NULL)
	    {
			if (strcmp(dir->d_name, ".") && strcmp(dir->d_name, ".."))
			{
				string nameOneClass(dir->d_name);
				nameClasses.push_back(nameOneClass);
//				printf("%s\n", dir->d_name);
			}
	    }
	    closedir(d);
	}

	sort(nameClasses.begin(), nameClasses.end()); // sort by the name
	int numClassTotal = nameClasses.size();  // 101 classes

	/* Process all the classes */
//	for (int c=60; c<61; c++)
	for (int c=0; c<numClassTotal; c++)
//	for (int c=numClassTotal-1; c>=0; c--)
	{
		cout << "Current class: " << c << ". " << nameClasses[c] << endl;

		const int64 startClass = getTickCount(); // computation time of one class

		string dirClass = inDir + nameClasses[c] + "/";

		// retrieve all the videos in the class folder
		DIR           *d_c;
		struct dirent *dir_c;
		d_c = opendir(dirClass.c_str());
		vector<string> nameSubVideos;

		if (d_c)
		{
			while ((dir_c = readdir(d_c)) != NULL)
			{
				if (strcmp(dir_c->d_name, ".") && strcmp(dir_c->d_name, ".."))
				{
					string nameOneSubVideo(dir_c->d_name);
					nameSubVideos.push_back(nameOneSubVideo);
//					printf("%s\n", dir_c->d_name);
				}
			}
			closedir(d_c);
		}

		sort(nameSubVideos.begin(), nameSubVideos.end()); // sort by the name
		int numSubVideoTotal = nameSubVideos.size();  // video #

		// Output information
		string outDirClass = outDir + nameClasses[c] + "/";
//		cout << outDirClass;

		if (!opendir(outDirClass.c_str())) // create the folder if not existed
		{
			mkdir(outDirClass.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		}

		/* Process all the videos */
//		for (int sv=146; sv<148; sv++)
		for (int sv=0; sv<numSubVideoTotal; sv++)
		{
			cout << sv << " " << flush;
			// process the video name
			vector<string> nameParse;
			split(nameParse, nameSubVideos[sv], is_any_of("."));

			string pathVideoIn = dirClass + nameSubVideos[sv]; // input
			string nameFlow = nameParse[0] + "_flow";
			string pathVideoOut = outDirClass + nameFlow + "." + nameParse[1]; // output

			if(!type.compare("CPU"))
			{
				VideoCapture reader(pathVideoIn);

				// video information
				double FPS = reader.get(CV_CAP_PROP_FPS)/numStep; // frame rate

				// read the first frame
				Mat framePrvs;
				reader.read(framePrvs);

				// video writer initialization
				VideoWriter writer(pathVideoOut, CV_FOURCC('X', 'V', 'I', 'D'), FPS, framePrvs.size());

				if ( !writer.isOpened() ) //if not initialize the VideoWriter successfully, exit the program
				{
					cout << "ERROR: Failed to write the video" << endl;
					return -1;
				}

				GpuMat g_flow(framePrvs.size(), CV_32FC2);

				int countFrame = 0;
				int countStep = 0;

//				const int64 start = getTickCount();

				// read the next frame & calculate the optical flow
				for (;;)
				{
					Mat frameNext;
					if (!reader.read(frameNext))
					{
						//cout << "Cannot read the frame" << endl;
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

						if(!methodOF.compare("Brox"))
						{brox->calc(framePrvs_f, frameNext_f, g_flow);}
						else if(!methodOF.compare("TVL1"))
						{tvl1->calc(framePrvs_f, frameNext_f, g_flow);}

						// Color coding
						Mat flow_out;
						showFlow(nameFlow + " (" + methodOF + ")", g_flow, flow_out);

						// show the frames & result flow map
						Mat imgDisplay;
						hconcat(framePrvs, frameNext, imgDisplay);
						hconcat(imgDisplay, flow_out, imgDisplay);

						imshow(nameFlow + " (" + methodOF + ")", imgDisplay);

						waitKey(3);
//						if (waitKey(3) > 0)
//							break;

						// save video
						writer.write(flow_out);

						framePrvs = frameNext; // update the previous frame
						countStep = 0; // restart the counter
					}
				}

//				const double timeSec = (getTickCount() - start) / getTickFrequency();
//				cout << "Computation time for " + nameFlow + " (Brox): " << timeSec << " sec" << endl;
				destroyWindow(nameFlow + " (" + methodOF + ")");
			}
		}
		const double timeSec = (getTickCount() - startClass) / getTickFrequency();
		cout << "Computation time for " + nameClasses[c] + " (" + methodOF + "): " << timeSec/60.0 << " min. " << endl;

	}
	cout << "Finished!!!!!!" << endl;


////////////////////////////////////////////////////////////////////////
//    else if(!type.compare("GPU"))
//    {
//    	Ptr<cv::cudacodec::VideoReader> g_reader = cv::cudacodec::createVideoReader(nameVideo);
//    	// read the first frame
//    	GpuMat g_framePrvs;
//    	g_reader->nextFrame(g_framePrvs);
//
//    	GpuMat g_flow(g_framePrvs.size(), CV_32FC2);
//
//        // read the next frame & calculate the optical flow
//        for (;;)
//        {
////        	const int64 start = getTickCount();
//
//        	cvtColor(g_framePrvs, g_framePrvs, CV_BGR2GRAY); // convert to grayscale
//
//        	GpuMat g_frameNext;
//            if (!g_reader->nextFrame(g_frameNext))
//            {
//            	cout << "Cannot read the frame" << endl;
//            	break;
//            }
//
//       	    GpuMat framePrvs_f;
//       	    GpuMat frameNext_f;
//
//       	    g_framePrvs.convertTo(framePrvs_f, CV_32F, 1.0 / 255.0);
//       	    g_frameNext.convertTo(frameNext_f, CV_32F, 1.0 / 255.0);
//
//       	    brox->calc(framePrvs_f, frameNext_f, g_flow);
//
////            const double timeSec = (getTickCount() - start) / getTickFrequency();
////            cout << "Brox : " << timeSec << " sec" << endl;
//
//       	    Mat flow_out;
//       	    showFlow("Brox", g_flow, flow_out);
//	waitKey(3);
//        	if (waitKey(3) > 0)
//        		break;
//
//        	g_framePrvs = g_frameNext;
//
//        }
//    }

    return 0;
}
