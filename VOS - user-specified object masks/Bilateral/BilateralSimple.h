#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"
#include "Bilateral.h"

class BilateralSimple
{
public:
	std::vector<Mat> imgSrcArr;	 
	Mat bgModel, fgModel, unModel;	
	Mat grid, gridColor;	
	const int gridSize[6] = { 1,30,50,16,16,16 };	

public:
	BilateralSimple(std::vector<Mat> img);
	~BilateralSimple();
	void InitGmms(Mat& maskArr);
	void run(Mat&);
private:
	void initGrid();
	void constructGCGraph(GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, Mat&);
	void getGridPoint(int, const Point, int *, int, int, int);
	void getGridPoint(int, const Point, std::vector<int>&, int, int, int);
};

