#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  
	fgdSum = 1,  
	bgdSum = 2,  
	vIdx = 3,   
};

class Bilateral
{
public:
	std::vector<Mat> imgSrcArr, dstImgSrcArr;
	std::vector<Mat> bgModelArr, fgModelArr, unModelArr;	
	Mat grid,gridColor, gridProbable;	
	const int gridSize[6] = {13,30,50,16,16,16};	//grid size:[t,x,y,r,g,b]
	std::vector<Vec3f> allSamples;

public:
	Bilateral(std::vector<Mat> img);
	~Bilateral();
	void InitGmms(std::vector<Mat>& , int*);
	void run(std::vector<Mat>& );
	void getGmmProMask(std::vector<Mat>&);
	void getKeyProMask(std::vector<Mat>&);
	void getTotalProMask(std::vector<Mat>&);
	void savePreImg(GCGraph<double>& graph);
private:
	void initGrid();
	void constructGCGraph(GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, std::vector<Mat>& );
	void getGridPoint(int , const Point , int *, int , int , int );
	void getGridPoint(int , const Point , std::vector<int>& , int , int , int );
	void getColor();
};

#endif