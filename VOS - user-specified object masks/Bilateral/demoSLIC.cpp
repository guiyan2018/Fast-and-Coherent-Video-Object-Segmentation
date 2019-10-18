#include <cstdlib>
#include <cstdio>
#include <ctime>

#include "slic.h"
#include "opencv2/opencv.hpp"

using namespace std;

int main()
{
	cv::Mat img, result;

	img = cv::imread("00003.jpg");
	int numSuperpixel = 3000;

	SLIC slic;
	
	clock_t clock_begin, clock_end;
	clock_begin = clock();
	
	slic.GenerateSuperpixels(img, numSuperpixel);
	
	clock_end = clock();
	printf("time elapsed: %f (ms), for img size: %dx%d\n", (float)(clock_end - clock_begin) / CLOCKS_PER_SEC * 1000, img.rows, img.cols);

	if (img.channels() == 3) 
		result = slic.GetImgWithContours(cv::Scalar(0, 0, 255));
	else
		result = slic.GetImgWithContours(cv::Scalar(128));

	char result_name[128];
	sprintf_s(result_name, 128, "result_%d.bmp", numSuperpixel);
	cv::imwrite(result_name, result);
	cv::imshow("result", result);
	cv::waitKey(0);

	return 0;
}
