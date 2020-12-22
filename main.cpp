#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "faceDetect.h"

#define DISP_WINNAME "camera"
#define QUIT_KEY 'q'
#define CAMID    0


using namespace std;

int main(int argc, const char* argv[]) {

	double ftick, etick;
	double ticksPerUs = cv::getTickFrequency() / 1000000;

	//string file = "D:/software/cmake_libtorch_tools/example_release/data.txt";
	string  modelpath = "D:/software/cmake_libtorch_tools/example_release/facebox_ir.pt";
	//加载模型
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(modelpath);
	module->to(at::kCUDA);
	assert(module != nullptr);

	vector <defaultbox> boxes;
	Facedetect F;
	//读取数组
	//F.read_txt(boxes_v1, file);
	F.computDefaultbox(boxes);
	assert(boxes.size() != 0);
	cout << "boxes_v1:"<<boxes.size()<<endl;

	cv::VideoCapture camera(CAMID);

	if (!camera.isOpened()) {
		std::cerr << "failed to camera" << std::endl;
		return 1;
	}

	cv::namedWindow(DISP_WINNAME, cv::WINDOW_AUTOSIZE);
	cv::Mat frame;

	do {
		camera >> frame;
		if (!frame.data)
		{
			std::cerr << "Capture video failed" << std::endl;
			break;
		}
		vector<float> finalBox;
		cv::Mat frame_gray;
		cv::cvtColor(frame.clone(),frame_gray,cv::COLOR_BGR2GRAY);
		ftick = cv::getCPUTickCount();
		int note;
		note = F.detect(frame_gray, finalBox, module, boxes);
		cout << "frame.cols" << frame.cols << endl;
		cout << "frame.rows"<<frame.rows<<endl;
		if (note == -1) {
			std::cout << "Error: Face detect failed" << std::endl;
		}
		etick = cv::getCPUTickCount();
		for (int i = 0; i < finalBox.size(); i++) {

			int x1 = int(finalBox[0]);
			int y1 = int(finalBox[1]);
			int x2 = int(finalBox[2]);
			int y2 = int(finalBox[3]);

			cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), cv::Scalar(0, 255, 255), 3);
		}
		std::cout << "total detected: " << finalBox.size() << "faces. used" << (etick - ftick) / ticksPerUs << "us" << std::endl;
		cv::imshow(DISP_WINNAME, frame);
		finalBox.clear();
	} while (QUIT_KEY != cv::waitKey(1));
	boxes.clear();

	return 0;
}