#ifndef _faceDetect_H_
#define _faceDetect_H_

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

//����Bbox
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


//���������
class Facedetect
{
private:
	float nms_threshold;//nms��ֵĬ�ϴ�С0.5���ɵ�
	float conf_threshold;//�����Ŷ���ֵĬ�ϴ�С0.8���ɵ�
	int IMAGE_SIZE;//д��
	int n;//д��
	int	m;//д��

public:
	Facedetect();
	//��ȡ�����ļ�
	void read_txt(std::vector<float>&dataset, std::string &file);
	//�ѿ�����
	void GetCartesianProduct(vector<vector<double>> &src, vector<vector<double>>&res, int nLyr, vector<double>&tmp);
	//����defaultbox
	void computDefaultbox(vector<defaultbox>&boxes);
	//softmax����
	void softmax(vector<vector<float>> input, vector<vector<float>> &output);
	//padding������Ĭ�ϲ�0
	void frame_pad(cv::Mat input, cv::Mat &output);
	//���÷�������������nms
	static bool sort_score(Bbox box1, Bbox box2);
	//�������������������������ʱɸѡ����
	static bool sort_area(Bbox box1, Bbox box2);
	//iou����
	float iou(Bbox box1, Bbox box2);
	//nms����
	void nms(std::vector<Bbox> &boundingBox_, float &overlap_threshold);
	//���붯̬�ڴ�
	void new_data(float** &boxes_decode, int &n);
	//ɾ���ڴ�
	void delete_data(float** &boxes_decode);
	//ɸѡ����
	void maxBox(vector<Bbox> &nms_data, vector<float> &finalBox, int image_width, int image_height);
	//���������
	int detect(cv::Mat &image, vector<float> &finalBox, std::shared_ptr<torch::jit::script::Module> &module, vector <defaultbox> boxes);

};

#endif



