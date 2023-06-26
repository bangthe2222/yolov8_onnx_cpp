#include <iostream>
#include<opencv2/opencv.hpp>

#include<math.h>
#include "include/yolov8_onnx.h"
#include <time.h>

using namespace std;
using namespace cv;
using namespace dnn;

template<typename _Tp>
int yolov8_onnx(_Tp& cls, Mat& img)
{
	vector<Scalar> color;
	srand(0);
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}
	vector<OutputSeg> result;
	if (cls.OnnxDetect(img, result)) {
		DrawPred(img, result, cls._className, color);
	}
	else {
		cout << "Detect Failed!" << endl;
	}
	// system("pause");
	return 0;
}

int main() {

	string detect_model_path = "D:/MAIN_DOCUMENTS/PROJECT/OPENCV_CPP/yolov8-opencv-onnxruntime-cpp/models/yolov8n.onnx";
	Yolov8Onnx task_detect_onnx;
	task_detect_onnx.ReadModel(detect_model_path, false);
	VideoCapture cap;
	cap.open(0);
	Mat frame;
	if (!cap.isOpened()){
		std::cout<<"Error open camera"<<std::endl;
	}
	while (true)
	{
		int64 start = cv::getTickCount();
		cap.read(frame);
		yolov8_onnx(task_detect_onnx, frame);
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		std::cout<<"fps: "<< fps<<std::endl;
		cv::imshow("image", frame);
		cv::waitKey(1);
	}
	
	return 0;
}


