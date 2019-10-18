#include "Bilateral.h"
#include <iostream>
#include <fstream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "libDenseCRF/densecrf.h"
#include "libDenseCRF/util.h"
#include <string>
#include <vector>
#include "BilateralSimple.h"
#include <sstream>
#include <io.h>
using namespace std;
using namespace cv;

/*dynamic appearance model with reliability measurements and higher-order potential*/

Bilateral::Bilateral(std::vector<Mat> img) :
	imgSrcArr(img) {
	initGrid();
}

Bilateral::~Bilateral()
{
	imgSrcArr.clear();
	bgModelArr.clear();
	fgModelArr.clear();
	grid.release();
}

//Reliable and dynamic appearance model
void Bilateral::InitGmms(std::vector<Mat>& maskArr, int* index)
{
	double _time = static_cast<double>(getTickCount());

	int maskSize = maskArr.size();
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	int point[6] = { 0,0,0,0,0,0 };

	for (int t = 0; t < maskSize; t++) {
		for (int x = 0; x < xSize; x++)
		{
			for (int y = 0; y < ySize; y++)
			{				
				if (maskArr[t].at<uchar>(x, y) == GC_BGD) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 3;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_FGD) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 3;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_PR_FGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[fgdSum] += 1;
				}
				else if (maskArr[t].at<uchar>(x, y) == GC_PR_BGD/*GC_FGD*/) {
					getGridPoint(index[t], Point(x, y), point, tSize, xSize, ySize);
					grid.at<Vec< int, 4 > >(point)[bgdSum] += 1;
				}
			}
		}
	}

	std::vector<Vec3f> bgdSamples;    
	std::vector<Vec3f> fgdSamples;   
	
	std::vector<std::vector<double> > bgdWeight(gridSize[0]);   
	std::vector<std::vector<double> > fgdWeight(gridSize[0]);   

	std::vector<Vec3f> unSamples;   
	std::vector<double>  unWeight;
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
				
								int bgdcount = grid.at<Vec< int, 4 > >(point)[bgdSum];
								int fgdcount = grid.at<Vec< int, 4 > >(point)[fgdSum];

								Vec3f color = gridColor.at<Vec3f>(point);
								if (bgdcount>0 && fgdcount>0)
								{
									
									unSamples.push_back(color);
									unWeight.push_back(1);
								}
								else {
									if (bgdcount > (pixCount / 2)) {
										bgdSamples.push_back(color);
										for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
											double weight = pixCount*exp(-2 * (t - tGmm)*(t - tGmm))*(bgdcount / (fgdcount + bgdcount));
											bgdWeight[tGmm].push_back(weight);
										}
									}
									if (fgdcount >(pixCount / 2)) {

										fgdSamples.push_back(color);
										for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
											double weight = pixCount*exp(-2 * (t - tGmm)*(t - tGmm))*(fgdcount / (fgdcount + bgdcount));
											fgdWeight[tGmm].push_back(weight);//weight
										}
									}

								}
								allSamples.push_back(color);
							}
						}
					}
				}
			}
		}
	}




	for (int tGmm = 0; tGmm < gridSize[0]; tGmm++) {
		Mat bgModel, fgModel;
		GMM bgdGMM(bgModel), fgdGMM(fgModel);

		const int kMeansItCount = 10;  
		const int kMeansType = KMEANS_PP_CENTERS;  
		Mat bgdLabels, fgdLabels;
				
		Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
		kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		
		Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
		kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
				
		bgdGMM.initLearning();
				
		for (int i = 0; i < (int)bgdSamples.size(); i++)
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[tGmm][i]);
		bgdGMM.endLearning();
		
		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[tGmm][i]);
		fgdGMM.endLearning();

		for (int times = 0; times < 5; times++)
		{			
			for (int i = 0; i < (int)bgdSamples.size(); i++) {
				Vec3d color = bgdSamples[i];
				bgdLabels.at<int>(i, 0) = bgdGMM.whichComponent(color); 
			}

			for (int i = 0; i < (int)fgdSamples.size(); i++) {
				Vec3d color = fgdSamples[i];
				fgdLabels.at<int>(i, 0) = fgdGMM.whichComponent(color);
			}
			
			bgdGMM.initLearning();
			for (int i = 0; i < (int)bgdSamples.size(); i++)
				bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], bgdWeight[tGmm][i]);
			bgdGMM.endLearning();

			fgdGMM.initLearning();
			for (int i = 0; i < (int)fgdSamples.size(); i++)
				fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], fgdWeight[tGmm][i]);
			fgdGMM.endLearning();
		}

		bgModelArr.push_back(bgModel);
		fgModelArr.push_back(fgModel);


	
		for (int i = 0; i < (int)bgdSamples.size(); i++) {
			Vec3d color = bgdSamples[i];
			double b = bgdGMM(color), f = fgdGMM(color);
			if (fgdGMM.bigThan1Cov(color) || (b < f)) {
				unSamples.push_back(color);
				unWeight.push_back(1);
			}
		}
		for (int i = 0; i < (int)fgdSamples.size(); i++) {
			Vec3d color = fgdSamples[i];
			double b = bgdGMM(color), f = fgdGMM(color);
			if (bgdGMM.bigThan1Cov(color) || (b > f)) {
				unSamples.push_back(color);
				unWeight.push_back(1);
			}
		}

			Mat unModel;
			GMM unGMM(unModel);
			Mat unLabels;
			Mat _unSamples((int)unSamples.size(), 3, CV_32FC1, &unSamples[0][0]);
			kmeans(_unSamples, GMM::componentsCount, unLabels,
				TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
			unGMM.initLearning();
			for (int i = 0; i < (int)unSamples.size(); i++)
				unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], unWeight[i]);
			unGMM.endLearning();
			for (int times = 0; times < 3; times++)
			{
				for (int i = 0; i < (int)unSamples.size(); i++) {
					Vec3d color = unSamples[i];
					unLabels.at<int>(i, 0) = unGMM.whichComponent(color);
				}
				unGMM.initLearning();
				for (int i = 0; i < (int)unSamples.size(); i++)
					unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], unWeight[i]);
				unGMM.endLearning();
			}
			unModelArr.push_back(unModel);
		
		

	}

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("Time for dynamic appearance modeling:%f\n", _time);
}

//Video data resampling by using bilateral grid
void Bilateral::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(4), Scalar(0, 0, 0, -1));
	Mat C(6, gridSize, CV_32FC(3), Scalar::all(0));
	Mat P(6, gridSize, CV_32FC(3), Scalar::all(0));
	grid = L;gridColor = C;gridProbable = P;
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int x = 0; x < xSize; x++)
		{
			//#pragma omp parallel for
			for (int y = 0; y < ySize; y++)
			{
				int tNew = gridSize[0] * t / tSize;
				int xNew = gridSize[1] * x / xSize;
				int yNew = gridSize[2] * y / ySize;
				Vec3b color = (Vec3b)imgSrcArr[t].at<Vec3b>(x, y);
				int rNew = gridSize[3] * color[0] / 256;
				int gNew = gridSize[4] * color[1] / 256;
				int bNew = gridSize[5] * color[2] / 256;
				int point[6] = { tNew,xNew,yNew,rNew,gNew,bNew };
				int count = ++(grid.at<Vec< int, 4 > >(point)[pixSum]);
				Vec3f colorMeans = gridColor.at<Vec3f>(point);
				colorMeans[0] = colorMeans[0] * (count - 1.0) / (count + 0.0) + color[0] / (count + 0.0);
				colorMeans[1] = colorMeans[1] * (count - 1.0) / (count + 0.0) + color[1] / (count + 0.0);
				colorMeans[2] = colorMeans[2] * (count - 1.0) / (count + 0.0) + color[2] / (count + 0.0);
				gridColor.at<Vec3f>(point) = colorMeans;
			}
		}
	}
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("Time for bilateral grid construction:%f\n", _time);
}

//Parameter: beta
static double calcBeta(const Mat& img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{ 
			Vec3d color = img.at<Vec3b>(y, x);
			if (x > 0)  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff); 
			}
			if (y > 0 && x > 0) // upleft  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y > 0) // up  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y > 0 && x < img.cols - 1) // upright  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2)); 

	return beta;
}

//Reading files
void getFiles(string path, string exd, vector<string>& files)
{
	intptr_t   hFile = 0;
	struct _finddata_t fileinfo;
	string pathName, exdName;

	if (0 != strcmp(exd.c_str(), ""))
	{
		exdName = "\\*." + exd;
	}
	else
	{
		exdName = "\\*";
	}

	if ((hFile = _findfirst(pathName.assign(path).append(exdName).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(pathName.assign(path).append("\\").append(fileinfo.name), exd, files);
			}
			else
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					files.push_back(pathName.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

//Build graph
void Bilateral::constructGCGraph(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	double bata = calcBeta(imgSrcArr[0]);
	int vtxCount = calculateVtxCount();
	int edgeCount = 2 * 256 * vtxCount;
	graph.create(vtxCount, edgeCount);

	Mat allLabels;
	const int kMeansItCount = 10;
	const int kMeansType = KMEANS_PP_CENTERS;
	int clusterNum = GMM::c2;
	Mat _allSamples((int)allSamples.size(), 3, CV_32FC1, &allSamples[0][0]);
	kmeans(_allSamples, GMM::c2, allLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
	double af = 0, ab = 0, bf = 0, bb = 0, cf = 0, cb = 0, df = 0, db = 0, ef = 0, eb = 0;
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int vtxId = graph.addVtx();
								grid.at<Vec< int, 4 > >(point)[vIdx] = vtxId;
								int label = allLabels.at<int>(grid.at<Vec< int, 4 > >(point)[vIdx], 0);
								double fSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double bSum = grid.at<Vec< int, 4 > >(point)[bgdSum];
								double pf = (bSum + 1.0) / (fSum + bSum + 2.0);
								double pb = (fSum + 1.0) / (fSum + bSum + 2.0);
								//parameter1 sum of P(Vj)
								if (label == 0) {
									af = af + (bSum + 1.0) / (fSum + bSum + 2.0);
									ab = ab + (fSum + 1.0) / (fSum + bSum + 2.0);
								}
								else if (label == 1) {
									bf = bf + (bSum + 1.0) / (fSum + bSum + 2.0);
									bb = bb + (fSum + 1.0) / (fSum + bSum + 2.0);
								}
								else if (label == 2) {
									cf = cf + (bSum + 1.0) / (fSum + bSum + 2.0);
									cb = cb + (fSum + 1.0) / (fSum + bSum + 2.0);
								}
								else if (label == 3) {
									df = df + (bSum + 1.0) / (fSum + bSum + 2.0);
									db = db + (fSum + 1.0) / (fSum + bSum + 2.0);
								}
								else if (label == 4) {
									ef = ef + (bSum + 1.0) / (fSum + bSum + 2.0);
									eb = eb + (fSum + 1.0) / (fSum + bSum + 2.0);
								}

							}

						}
					}
				}
			}
		}
	}
	double AF = 0, AB = 0, BF = 0, BB = 0, CF = 0, CB = 0, DF = 0, DB = 0, EF = 0, EB = 0;
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int label = allLabels.at<int>(grid.at<Vec< int, 4 > >(point)[vIdx], 0);
								double fSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double bSum = grid.at<Vec< int, 4 > >(point)[bgdSum];
								double pf = (bSum + 1.0) / (fSum + bSum + 2.0);
								double pb = (fSum + 1.0) / (fSum + bSum + 2.0);
								//parameter2 sum of (P(Vj)-para1/|c|)*(P(Vj)-para1/|c|)
								if (label == 0) {
									AF = AF + (pf - af / clusterNum)*(pf - af / clusterNum);
									AB = AB + (pb - ab / clusterNum)*(pb - ab / clusterNum);									 
								}
								else if (label == 1) {
									BF = BF + (pf - bf / clusterNum)*(pf - bf / clusterNum);
									BB = BB + (pb - bb / clusterNum)*(pb - bb / clusterNum);
								}
								else if (label == 2) {
									CF = CF + (pf - cf / clusterNum)*(pf - cf / clusterNum);
									CB = CB + (pb - cb / clusterNum)*(pb - cb / clusterNum);
								}
								else if (label == 3) {
									DF = DF + (pf - df / clusterNum)*(pf - df / clusterNum);
									DB = DB + (pb - db / clusterNum)*(pb - db / clusterNum);
								}
								else if (label == 4) {
									EF = EF + (pf - ef / clusterNum)*(pf - ef / clusterNum);
									EB = EB + (pb - eb / clusterNum)*(pb - eb / clusterNum);
								}
							}

						}
					}
				}
			}
		}
	}
	double sumF = AF + BF + CF + DF + EF;
	AF = AF / sumF;
	BF = BF / sumF;
	CF = CF / sumF;
	DF = DF / sumF;
	EF = EF / sumF;
	double sumB = AB + BB + CB + DB + EB;
	AB = AB / sumB;
	BB = BB / sumB;
	CB = CB / sumB;
	DB = DB / sumB;
	EB = EB / sumB;
	
	for (int t = 0; t < gridSize[0]; t++) {
		int gmmT = t;
		GMM bgdGMM(bgModelArr[gmmT]), fgdGMM(fgModelArr[gmmT]), unGMM(unModelArr[gmmT]);
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								double fSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double bSum = grid.at<Vec< int, 4 > >(point)[bgdSum];
								double pf = (bSum + 1.0) / (fSum + bSum + 2.0);
								double pb = (fSum + 1.0) / (fSum + bSum + 2.0);

								int vtxIdx = graph.addVtx();
								//grid.at<Vec< int, 4 > >(point)[vIdx] = vtxIdx;					
								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;
								double E1f, E1b;
								double unWeight;
								if ((bSum > pixCount) && fSum == 0) {
									E1f = 0;
									E1b = 9999;
								}
								else if (bSum == 0 && (fSum > pixCount)) {
									E1f = 9999;
									E1b = 0;
								}
								else {
									double bgd = bgdGMM(color);
									double fgd = fgdGMM(color);
									double un = unGMM(color);
									unWeight = 1.0 - (un / (bgd + fgd + un));
									double sumWeight = abs(bSum - fSum) / (bSum + fSum + 1.0);
									if (unGMM.bigThan1Cov(color) || (bgdGMM.smallThan2Cov(color) && fgdGMM.smallThan2Cov(color))) {
										bgd = fgd = 0.5;
									}
									gridProbable.at<Vec3f>(point)[0] = fgd / (bgd + fgd);


									gridProbable.at<Vec3f>(point)[1] = (fSum + 1.0) / (fSum + bSum + 2.0);
									fromSource = (-log(bgd / (bgd + fgd))*unWeight - log((bSum + 1.0) / (fSum + bSum + 2.0))*sumWeight)*sqrt(pixCount);
									toSink = (-log(fgd / (bgd + fgd))*unWeight - log((fSum + 1.0) / (fSum + bSum + 2.0))*sumWeight)*sqrt(pixCount);

									int label = allLabels.at<int>(grid.at<Vec< int, 4 > >(point)[vIdx], 0);
									double sf = 0, sb = 0;
									double saf, sab, sbf, sbb, scf, scb, sdf, sdb, sef, seb;
									if (label == 0) {
										sf = exp((0.5 / clusterNum)*AF);
										sb = exp((0.5 / clusterNum)*AB);
									}
									else if (label == 1) {
										sf = exp((0.5 / clusterNum)*BF);
										sb = exp((0.5 / clusterNum)*BB);
									}
									else if (label == 2) {
										sf = exp((0.5 / clusterNum)*CF);
										sb = exp((0.5 / clusterNum)*CB);
									}
									else if (label == 3) {
										sf = exp((0.5 / clusterNum)*DF);
										sb = exp((0.5 / clusterNum)*DB);
									}
									else if (label == 4) {
										sf = exp((0.5 / clusterNum)*EF);
										sb = exp((0.5 / clusterNum)*EB);
									}

									E1f = fromSource + (sf / clusterNum)*pf;
									E1b = toSink + (sb / clusterNum)*pb;
									gridProbable.at<Vec3f>(point)[2] = (fromSource) / (fromSource + toSink);
								}
								graph.addTermWeights(vtxIdx, E1f, E1b);

								for (int tN = t; tN > t - 2 && tN >= 0 && tN < gridSize[0]; tN--) {
									for (int xN = x; xN > x - 2 && xN >= 0 && xN < gridSize[1]; xN--) {
										for (int yN = y; yN > y - 2 && yN >= 0 && yN < gridSize[2]; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0 && rN < gridSize[3]; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0 && gN < gridSize[4]; gN--) {
													for (int bN = b; bN > b - 2 && bN >= 0 && bN < gridSize[5]; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
														if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
															double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));
															double w = 1.0 * e * sqrt(num);
															graph.addEdges(vtxIdx, vtxIdxNew, w, w);
														}
													}
												}
											}
										}
									}
								}


							}


						}
					}
				}
			}

		}
	}
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("Time for graph construction:%f\n", _time);
}

//Count the number of grid cells
int Bilateral::calculateVtxCount() {
	int count = 0;
	for (int t = 0; t < gridSize[0]; t++)
	{
		for (int x = 0; x < gridSize[1]; x++)
		{
			for (int y = 0; y < gridSize[2]; y++)
			{
				for (int r = 0; r < gridSize[3]; r++)
				{
					for (int g = 0; g < gridSize[4]; g++)
					{
						for (int b = 0; b < gridSize[5]; b++)
						{
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								count++;
							}
						}
					}
				}
			}
		}
	}
	return count;
}

//Max-flow algorithm
void Bilateral::estimateSegmentation(GCGraph<double>& graph, std::vector<Mat>& maskArr) {
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("Time for graph cut:%f\n", _time);

	double _time2 = static_cast<double>(getTickCount());
	int tSize = maskArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int y = 0; y < ySize; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
				if (graph.inSourceSegment(vertex))//foreground for '255'
					maskArr[t].at<uchar>(p.x, p.y) = 255;
				else//background for '0'
					maskArr[t].at<uchar>(p.x, p.y) = 0;
			}
		}
	}

	_time2 = (static_cast<double>(getTickCount()) - _time2) / getTickFrequency();
	printf("Time for gird to masks: %f\n", _time2);
}

//Get grid cells
void Bilateral::getGridPoint(int index, const Point p, int *point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.x / xSize;
	point[2] = gridSize[2] * p.y / ySize;
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p.x, p.y);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::getGridPoint(int index, const Point p, std::vector<int>& point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.y / xSize;
	point[2] = gridSize[2] * p.x / ySize;
	Vec3b color = (Vec3b)imgSrcArr[index].at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

//Computing average color of each grid cell
void Bilateral::getColor() {
	
	for (int r = 0; r < gridSize[3]; r++) {
		for (int g = 0; g < gridSize[4]; g++) {
			for (int b = 0; b < gridSize[5]; b++) {
				Vec3b color;
				color[0] = (r * 256 + 256 / 2) / gridSize[3];
				color[1] = (g * 256 + 256 / 2) / gridSize[4];
				color[2] = (b * 256 + 256 / 2) / gridSize[5];
			
				for (int t = 0; t < gridSize[0]; t++) {
					for (int x = 0; x < gridSize[1]; x++) {
						for (int y = 0; y < gridSize[2]; y++) {

							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] != -1) {
								Vec3f colorM = gridColor.at<Vec3f>(point);
								
							}
						}
					}
				}
			}
		}
	}
}

void Bilateral::getGmmProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	int tSize = maskArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		
		for (int x = 0; x < xSize; x++)
		{
			for (int y = 0; y < ySize; y++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				float probable = gridProbable.at<Vec3f>(point)[2];
				

				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
		
	}
}

void Bilateral::getKeyProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	int tSize = maskArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int y = 0; y < ySize; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				float probable = gridProbable.at<Vec3f>(point)[1];
				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
	}
}

//Probability masks
void Bilateral::getTotalProMask(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	int tSize = maskArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int y = 0; y < ySize; y++)
		{
#pragma omp parallel for
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				float probable = gridProbable.at<Vec3f>(point)[2];
				maskArr[t].at<uchar>(p.x, p.y) = (uchar)(probable * 255);
			}
		}
	}
}

//Run graphCut
void Bilateral::run(std::vector<Mat>& maskArr) {
	for (int i = 0; i < imgSrcArr.size(); i++)
	{
		Mat mask = Mat::zeros(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC1);
		maskArr.push_back(mask);
	}
	GCGraph<double> graph;

	constructGCGraph(graph);

	estimateSegmentation(graph, maskArr);
	savePreImg(graph);
}

//Save over-segmentation results for each video frame
void Bilateral::savePreImg(GCGraph<double>& graph) {
	int tSize = imgSrcArr.size();
	int xSize = imgSrcArr[0].rows;
	int ySize = imgSrcArr[0].cols;
	Mat randColor(6, gridSize, CV_8UC3, Scalar::all(0));

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
								randColor.at<Vec3b>(point) = { (uchar)(rand() % 255),(uchar)(rand() % 255),(uchar)(rand() % 255) };
							}
						}
					}
				}
			}
		}
	}

	
	Mat preSegImg(imgSrcArr[0].rows, imgSrcArr[0].cols, CV_8UC3);
	for (int t = 0; t < tSize; t++) {
		
		
		for (int y = 0; y < ySize; y++)
		{
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				Vec3b colorMeans = randColor.at<Vec3b>(point);
				preSegImg.at<Vec3b>(x, y) = colorMeans;
				
			}
		}
		
	}
	randColor.release();
}