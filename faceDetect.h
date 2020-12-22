#ifndef _faceDetect_H_
#define _faceDetect_H_

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

//定义Bbox
typedef struct Bbox {
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	float area;
}Bbox;

typedef struct defaultbox {
	double a;
	double b;
	double c;
	double d;
	defaultbox(double a_, double b_, double c_, double d_) :
		a(a_), b(b_), c(c_), d(d_)
	{}
};


//人脸检测类
class Facedetect
{
private:
	float nms_threshold;//nms阈值默认大小0.5，可调
	float conf_threshold;//框置信度阈值默认大小0.8，可调
	int IMAGE_SIZE;//写死
	int n;//写死
	int	m;//写死

public:
	Facedetect();
	//读取数据文件
	void read_txt(std::vector<float>&dataset, std::string &file);
	//笛卡尔积
	void GetCartesianProduct(vector<vector<double>> &src, vector<vector<double>>&res, int nLyr, vector<double>&tmp);
	//计算defaultbox
	void computDefaultbox(vector<defaultbox>&boxes);
	//softmax函数
	void softmax(vector<vector<float>> input, vector<vector<float>> &output);
	//padding函数，默认补0
	void frame_pad(cv::Mat input, cv::Mat &output);
	//按得分排序函数，用于nms
	static bool sort_score(Bbox box1, Bbox box2);
	//按面积排序函数，用于输出多个框时筛选最大框
	static bool sort_area(Bbox box1, Bbox box2);
	//iou函数
	float iou(Bbox box1, Bbox box2);
	//nms函数
	void nms(std::vector<Bbox> &boundingBox_, float &overlap_threshold);
	//申请动态内存
	void new_data(float** &boxes_decode, int &n);
	//删除内存
	void delete_data(float** &boxes_decode);
	//筛选最大框
	void maxBox(vector<Bbox> &nms_data, vector<float> &finalBox, int image_width, int image_height);
	//检测主函数
	int detect(cv::Mat &image, vector<float> &finalBox, std::shared_ptr<torch::jit::script::Module> &module, vector <defaultbox> boxes);

};

#endif



