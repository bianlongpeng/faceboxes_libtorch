#include <iostream>
#include <string>
#include <memory>

#include "faceDetect.h"

Facedetect::Facedetect() {
	this->nms_threshold = 0.5;
	this->conf_threshold = 0.03;
	this->IMAGE_SIZE = 512;
	this->n = 4;
	this->m = 2;
}

void Facedetect::read_txt(std::vector<float>&dataset_set, std::string &file){
	std::ifstream f;
	f.open(file, std::ios::in);
	float tmp;
	for (int i = 0; i < 21824 * 4; i++)
	{
		f >> tmp;
		dataset_set.push_back(tmp);
	}

	f.close();
}

//笛卡尔积
void Facedetect::GetCartesianProduct(vector<vector<double>> &src, vector<vector<double>>&res, int nLyr, vector<double>&tmp)
{
	if (src.empty())
		return;
	if (nLyr < src.size() - 1) {
		for (int i = 0; i < src[nLyr].size(); ++i) {
			vector<double> vTmpData;
			for (int j = 0; j < tmp.size(); j++) {
				vTmpData.push_back(tmp[j]);
			}
			vTmpData.push_back(src[nLyr][i]);
			GetCartesianProduct(src, res, nLyr + 1, vTmpData);
		}
	}
	else if (nLyr == src.size() - 1) {
		for (int j = 0; j < src[nLyr].size(); j++) {
			tmp.push_back(src[nLyr][j]);
			res.push_back(tmp);
			tmp.pop_back();
		}
	}
}

void  Facedetect::computDefaultbox(vector<defaultbox>&boxes) {
	double steps[3] = { 0.03125,0.0625,0.125 };
	double sizes[3] = { 0.03125,0.25,0.5 };
	vector<vector<double>> aspect_ratios;
	aspect_ratios.push_back({ 1,2,4 });
	aspect_ratios.push_back({ 1 });
	aspect_ratios.push_back({ 1 });
	int feature_map_sizes[3] = { 32, 16, 8 };
	vector<vector<double>> density;
	density.push_back({ -3,-1,1,3 });
	density.push_back({ -1,1 });
	density.push_back({ 0 });

	for (size_t i = 0; i < 3; i++)
	{
		int fmsize = feature_map_sizes[i];

		vector<double>t1;
		vector<double>tmp;
		vector<vector<double>>src;
		vector<vector<double>>res;
		int nlayer = 0;
		for (size_t n = 0; n < fmsize; n++)
		{
			t1.push_back(n);
		}
		src.push_back(t1);
		src.push_back(t1);
		this->GetCartesianProduct(src, res, nlayer, tmp);
		t1.clear();
		tmp.clear();
		src.clear();

		for (size_t k = 0; k < res.size(); k++)
		{
			double w = res[k][0];
			double h = res[k][1];

			double cx = (w + 0.5)*steps[i];
			double cy = (h + 0.5)*steps[i];
			double s = sizes[i];
			for (size_t j = 0; j < aspect_ratios[i].size(); j++)
			{
				double ar = aspect_ratios[i][j];
				if (i == 0)
				{
					vector<vector<double>>out;
					src.push_back(density[j]);
					src.push_back(density[j]);
					int mlayer = 0;
					this->GetCartesianProduct(src, out, mlayer, tmp);
					for (size_t m = 0; m < out.size(); m++)
					{
						double dx = out[m][0];
						double dy = out[m][1];
						boxes.push_back(defaultbox(cy + dy / 8.*s*ar, cx + dx / 8.*s*ar, s*ar, s*ar));

					}
					tmp.clear();
					src.clear();
					out.clear();
				}
				else
				{
					boxes.push_back(defaultbox(cy, cx, s*ar, s*ar));
				}
			}
		}
		res.clear();
	}
}




void Facedetect::softmax(vector<vector<float>> input, vector<vector<float>> &output) {
	for (int i = 0; i<input.size(); i++) {
		//cout<<i<<endl;
		vector<float> temp;
		temp.push_back(exp(input[i][0]) / (exp(input[i][0]) + exp(input[i][1])));
		temp.push_back(exp(input[i][1]) / (exp(input[i][0]) + exp(input[i][1])));
		output.push_back(temp);
		temp.clear();
	}
}

void Facedetect::frame_pad(cv::Mat input, cv::Mat &output) {
	int w = input.cols;
	int h = input.rows;
	int borderType = cv::BORDER_CONSTANT;
	int top;
	int bottom;
	int left;
	int right;
	if (w >= h) {
		top = int((w - h) / 2);
		bottom = int((w - h) / 2);
		left = 0;
		right = 0;
	}
	else
	{
		top = 0;
		bottom = 0;
		left = int((h - w) / 2);
		right = int((h - w) / 2);
	}

	copyMakeBorder(input, output, top, bottom, left, right, borderType);
}

bool Facedetect::sort_score(Bbox box1, Bbox box2) {
	return box1.score > box2.score ? true : false;
}

bool Facedetect::sort_area(Bbox box1, Bbox box2) {
	return box1.area < box2.area ? true : false;
}

float Facedetect::iou(Bbox box1, Bbox box2) {
	int x1 = max(box1.x1, box2.x1);
	int y1 = max(box1.y1, box2.y2);
	int x2 = min(box1.x2, box2.x2);
	int y2 = min(box1.y2, box2.y2);
	int w = max(0, x2 - x1 + 1);
	int h = max(0, y2 - y1 + 1);
	float over_area = w*h;
	float box1_area = (box1.x2 - box1.x1)*(box1.y2 - box1.y1);
	float box2_area = (box2.x2 - box2.x1)*(box2.y2 - box2.y1);
	return over_area / (box1_area + box2_area - over_area);
}

void Facedetect::nms(std::vector<Bbox> &boundingBox_, float &overlap_threshold) {
	if (boundingBox_.empty()) {
		return;
	}
	//对各个候选框根据score大小进行升序排列
	std::sort(boundingBox_.begin(), boundingBox_.end(), this->sort_score);
	float IOU = 0;
	float maxX = 0;
	float maxY = 0;
	float minX = 0;
	float minY = 0;
	vector<int> vPick;
	int nPick = 0;
	multimap<float, int> vScores;//存放升序排列后的score和对应的序号
	const int num_boxes = boundingBox_.size();
	vPick.resize(num_boxes);
	for (int i = 0; i < num_boxes; ++i) {
		vScores.insert(pair<float, int>(boundingBox_[i].score, i));
	}
	while (vScores.size() > 0)
	{
		int last = vScores.rbegin()->second;//反向迭代器，获得vScores序列的最后那个序列号
		vPick[nPick] = last;
		nPick += 1;
		for (multimap<float, int>::iterator it = vScores.begin(); it != vScores.end();)
		{
			int it_idx = it->second;
			maxX = max(boundingBox_.at(it_idx).x1, boundingBox_.at(last).x1);
			maxY = max(boundingBox_.at(it_idx).y1, boundingBox_.at(last).y1);
			minX = min(boundingBox_.at(it_idx).x2, boundingBox_.at(last).x2);
			minY = min(boundingBox_.at(it_idx).y2, boundingBox_.at(last).y2);
			//转换成了两个边界框相较区域的边长
			maxX = ((minX - maxX) > 0) ? (minX - maxX) : 0;
			maxY = ((minY - maxY) > 0) ? (minY - maxY) : 0;
			IOU = (maxX * maxY) / (boundingBox_.at(it_idx).area + boundingBox_.at(last).area - maxX*maxY);

			if (IOU > overlap_threshold)
			{
				it = vScores.erase(it);
			}
			else
			{
				it++;
			}
		}
	}

	vPick.resize(nPick);
	vector<Bbox> tmp_;
	tmp_.resize(nPick);
	for (int i = 0; i < nPick; i++)
	{
		tmp_[i] = boundingBox_[vPick[i]];
	}
	boundingBox_ = tmp_;
}

void Facedetect::new_data(float** &boxes_decode, int &n) {
	boxes_decode = new float*[21824];
	for (int i = 0; i < 21824; i++)
	{
		boxes_decode[i] = new float[n];
	}
}

void Facedetect::delete_data(float** &boxes_decode) {
	for (int i = 0; i < 21824; i++)
	{
		delete boxes_decode[i];
	}
	delete boxes_decode;
}

void Facedetect::maxBox(vector<Bbox> &nms_data, vector<float> &finalBox, int image_width, int image_height) {
	if (nms_data.size() > 2) {
		std::sort(nms_data.begin(), nms_data.end(), this->sort_area);//从大到小
	}
	if (image_width >= image_height){
		finalBox.push_back(nms_data[0].x1*image_width);
		finalBox.push_back(nms_data[0].y1*image_width - (image_width-image_height)/2);
		finalBox.push_back(nms_data[0].x2*image_width);
		finalBox.push_back(nms_data[0].y2*image_width - (image_width-image_height)/2);
	}
	else
	{
		finalBox.push_back(nms_data[0].x1*image_height - (image_height - image_width) / 2);
		finalBox.push_back(nms_data[0].y1*image_height);
		finalBox.push_back(nms_data[0].x2*image_height - (image_height - image_width) / 2);
		finalBox.push_back(nms_data[0].y2*image_height);
	}

}

int Facedetect::detect(cv::Mat &image, vector<float> &finalBox, std::shared_ptr<torch::jit::script::Module> &module, vector <defaultbox> boxes) {
	std::cout << "This is FaceDetect mudule" << endl;
	//判断图片是否为空
	if (image.empty()) {
		std::cout << "Error: image is empty" << endl;
		return -1;
	}
	//判断图片格式是否为三通道
	if (image.type() != CV_8UC1) {
		std::cout << "Error: image channels:" << image.channels() << endl;
		return -1;
	}
	int width = image.cols;
	int height = image.rows;

	//vector<float> temp_boxes;
	//vector <vector<float>> boxes_v2;
	////float boxes_array[21824][4];
	//float** boxes_array;
	//this->new_data(boxes_array, this->n);

	//for (int i = 0; i < 21824; i++) {
	//	temp_boxes.push_back(boxes_v1[i * 4 + 0]);
	//	temp_boxes.push_back(boxes_v1[i * 4 + 1]);
	//	temp_boxes.push_back(boxes_v1[i * 4 + 2]);
	//	temp_boxes.push_back(boxes_v1[i * 4 + 3]);
	//	boxes_v2.push_back(temp_boxes);
	//	temp_boxes.clear();
	//}
	////boxes_v1.clear();
	//for (int i = 0; i < 21824; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		boxes_array[i][j] = boxes_v2[i][j];
	//	}
	//}
	//boxes_v2.clear();


	//输入数据预处理	
	this->frame_pad(image, image);
	cv::resize(image, image, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
	image.convertTo(image, CV_32F);
	//image.convertTo(image, CV_32F, 1.0 / 255.0);
	auto img_tensor = torch::from_blob(image.data, { 1, image.rows, image.cols, 1 });
	img_tensor = img_tensor.permute({ 0, 3, 1, 2 });
	//img_tensor = img_tensor.toType(torch::kFloat);
	//img_tensor = img_tensor.div(255);
	//std::cout << "Image process success! " << endl;

	//把模型和图片放在cuda计算
	//module->to(at::kCUDA);
	img_tensor = img_tensor.to(at::kCUDA);

	//std::cout << "Image and module put on Cuda success!" << endl;
	//cout << "image_tensor:" << img_tensor.type << endl;

	//前向计算
	//auto img_tensor_tmp = torch::autograd::make_variable(img_tensor, false);//不需要梯度
	torch::jit::Stack outputs = module->forward({ img_tensor }).toTuple()->elements();
	//std::cout << "inference success!" << std::endl;
	if (outputs.empty()) {
		std::cout << "Error: module output is empty" << endl;
		return -1;
	}

	//转cpu
	auto output1 = outputs[0].toTensor().to(at::kCPU);
	auto output2 = outputs[1].toTensor().to(at::kCPU);
	auto loc_outputss = output1.accessor<float, 3>();
	auto conf_outputss = output2.accessor<float, 3>();
	//cout << "put output on cpu success!" << endl;
	cout << "loc:"<<loc_outputss[0][0][0] << endl;
	cout << "conf:" << conf_outputss[0][0][0] << endl;
	//解码
	//float loc_preds_array[21824][4];
	//float conf_preds_array[21824][2];
	//动态申请内存
	float** loc_preds_array;
	float** conf_preds_array;
	this->new_data(loc_preds_array, this->n);
	this->new_data(conf_preds_array, this->m);

	for (int i = 0; i < 21824; i++) {
		loc_preds_array[i][0] = loc_outputss[0][i][0];
		loc_preds_array[i][1] = loc_outputss[0][i][1];
		loc_preds_array[i][2] = loc_outputss[0][i][2];
		loc_preds_array[i][3] = loc_outputss[0][i][3];
		conf_preds_array[i][0] = conf_outputss[0][i][0];
		conf_preds_array[i][1] = conf_outputss[0][i][1];
	}
	//cout << "decode data phase1 success!" << endl;


	//float boxes_decode[21824][4];
	//float boxes_decode_[21824][4];
	float** boxes_decode;
	float** boxes_decode_;
	this->new_data(boxes_decode, this->n);
	this->new_data(boxes_decode_, this->n);
	float variances[2] = { 0.1, 0.2 };
	vector<int> idx;
	for (int i = 0; i < 21824; i++) {

		boxes_decode[0][0] = loc_preds_array[0][0] * variances[0] * boxes[0].c + boxes[0].a;
		boxes_decode[i][0] = loc_preds_array[i][0] * variances[0] * boxes[i].c+ boxes[i].a;
		boxes_decode[i][1] = loc_preds_array[i][1] * variances[0] * boxes[i].d + boxes[i].b;
		boxes_decode[i][2] = exp(loc_preds_array[i][2] * variances[1]) * boxes[i].c;
		boxes_decode[i][3] = exp(loc_preds_array[i][3] * variances[1]) * boxes[i].d;

		boxes_decode_[i][0] = boxes_decode[i][0] - boxes_decode[i][2] / 2.;
		boxes_decode_[i][1] = boxes_decode[i][1] - boxes_decode[i][3] / 2.;
		boxes_decode_[i][2] = boxes_decode[i][0] + boxes_decode[i][2] / 2.;
		boxes_decode_[i][3] = boxes_decode[i][1] + boxes_decode[i][3] / 2.;

		if (conf_preds_array[i][1] > conf_preds_array[i][0]) {
			idx.push_back(i);
		}
	}
	this->delete_data(loc_preds_array);
	this->delete_data(boxes_decode);
	//this->delete_data(boxes_array);
	//cout << "decode data phase2 success!" <<endl ;
	//cout<<idx.size()<<endl;

	vector<float> loc_temp;
	vector<float> conf_temp;
	vector<vector<float>> loc;
	vector<vector<float>> conf;
	vector<vector<float>> conf_output;
	for (int i = 0; i<idx.size(); i++) {
		int id = idx[i];
		loc_temp.push_back(boxes_decode_[id][0]);
		loc_temp.push_back(boxes_decode_[id][1]);
		loc_temp.push_back(boxes_decode_[id][2]);
		loc_temp.push_back(boxes_decode_[id][3]);
		loc.push_back(loc_temp);
		conf_temp.push_back(conf_preds_array[id][0]);
		conf_temp.push_back(conf_preds_array[id][1]);
		conf.push_back(conf_temp);
		loc_temp.clear();
		conf_temp.clear();
	}
	this->delete_data(boxes_decode_);
	this->delete_data(conf_preds_array);

	this->softmax(conf, conf_output);//compute confidence
	if (conf_output.empty()) {
		std::cout << conf.size() << endl;
		std::cout << "Error: softmax conf_output is empty" << endl;
		return -1;
	}

	Bbox bbox;
	vector<Bbox> nms_data;
	for (int i = 0; i<idx.size(); i++)
	{
		bbox.x1 = float(loc[i][0]);
		bbox.y1 = float(loc[i][1]);
		bbox.x2 = float(loc[i][2]);
		bbox.y2 = float(loc[i][3]);
		bbox.score = conf_output[i][1];
		bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
		if (bbox.score > this->conf_threshold) {
			nms_data.push_back(bbox);
		}
	}

	this->nms(nms_data, this->nms_threshold);
	if (nms_data.empty()) {
		std::cout << "Error: nms_data is empty" << endl;
		return -1;
	}
	this->maxBox(nms_data, finalBox, width, height);
	if (finalBox.empty()) {
		std::cout << "Error: finalBox is empty" << endl;
		return -1;
	}
	nms_data.clear();
	idx.clear();
	loc.clear();
	conf_output.clear();
	conf.clear();

	return  0;
}