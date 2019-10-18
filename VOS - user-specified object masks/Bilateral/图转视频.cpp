#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <string>
#include <sstream>
#include <io.h>

using namespace::cv;
using namespace::std;

void getFiles(string path, string exd, vector<string>& files)
{
	//文件句柄
	intptr_t   hFile = 0;
	//文件信息
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
			//如果是文件夹中仍有文件夹,迭代之
			//如果不是,加入列表
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

int main() {

	vector<string> files;
	std::string name = "abear";
	std::string filePath = "C:\\Users\\zengz\\Desktop\\data\\" + name;
	getFiles(filePath, "bmp", files);
	
	VideoWriter videowriter;
	std::string	strVideowriter = "C:\\Users\\zengz\\Desktop\\data\\" + name + ".avi";
	Mat g_imgSrc = imread(files[0]);
	videowriter.open(strVideowriter, CV_FOURCC('D', 'I', 'V', 'X'),24, Size(g_imgSrc.cols, g_imgSrc.rows));

	int size = files.size();
	for (int i = 0; i < size; i++)
	{		
		g_imgSrc = imread(files[i]);
		videowriter << g_imgSrc;
	}

	videowriter.release();
}