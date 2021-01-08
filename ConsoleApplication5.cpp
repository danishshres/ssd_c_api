
#include "pch.h"
#include <iostream>
#include "tensorflow/c/c_api.h"
#include <codecvt>
#include <fstream>
#include <vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/types.hpp>
#include<opencv2/imgproc.hpp>

void DeallocateBuffer(void* data, size_t) {
	std::free(data);
}

TF_Buffer* ReadBufferFromFile(std::string file) {
	std::ifstream f(file, std::ios::binary);
	if (f.fail() || !f.is_open()) {
		return nullptr;
	}

	f.seekg(0, std::ios::end);
	const auto fsize = f.tellg();
	f.seekg(0, std::ios::beg);

	if (fsize < 1) {
		f.close();
		return nullptr;
	}

	char* data = static_cast<char*>(std::malloc(fsize));
	f.read(data, fsize);
	f.close();

	TF_Buffer* buf = TF_NewBuffer();
	buf->data = data;
	buf->length = fsize;
	buf->data_deallocator = DeallocateBuffer;
	return buf;
}

typedef struct {
	void* buffer;
	int32_t width;
	int32_t height;
	int32_t stride;
	int32_t bit_depth;
}ImageDataRecord;

class Network{
	
	public:
		void LoadGraph(std::string modelPath);
		void Run(cv::Mat  image_record);
		static void Deallocator(void* data, size_t length, void* arg);
		Network();
		~Network();
	private:
		TF_Session* session;
		TF_Graph* graph;
};

Network::Network()
{
	std::cout << "Created a new network";
}

void Network::LoadGraph(std::string modelPath)
{
	graph = TF_NewGraph();
	TF_Status* Status = TF_NewStatus();
	TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
	TF_Buffer* RunOpts = NULL;
	const char* tags = "serve";
	int ntags = 1;
	char* path = const_cast<char*>(modelPath.c_str());

	session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, path, &tags, ntags, graph, NULL, Status);
	if (TF_GetCode(Status) == TF_OK)
	{
		printf("Tensorflow 2x Model loaded OK\n");
	}
	else
	{
		printf("%s", TF_Message(Status));
	}
	TF_DeleteSessionOptions(SessionOpts);
	TF_DeleteStatus(Status);
}

void Network::Run(cv::Mat image)
{

	std::vector<TF_Output> 	input_tensors, output_tensors;
	std::vector<TF_Tensor*> input_values, output_values;

	//input tensor shape.
	int num_dims = 4;
	std::int64_t input_dims[4] = {1, image.rows, image.cols, 3}; //1 is number of batch, and 3 is the no of channels.
	int num_bytes_in = image.cols * image.rows * 3; //3 is the number of channels.
	
	TF_Output t0 = { TF_GraphOperationByName(graph, "StatefulPartitionedCall"),0};
	if (t0.oper == NULL)
		printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
	else
		printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

	size_t pos = 0;
	TF_Operation* oper;
	
	while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
		printf(TF_OperationName(oper));
		printf("\n");
	}
	input_tensors.push_back({TF_GraphOperationByName(graph, "serving_default_input_tensor"),0});
	input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));


	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),0});
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),2 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "StatefulPartitionedCall"),3 });
	output_values.push_back(nullptr);

	TF_Status* status = TF_NewStatus();
	TF_SessionRun(session, nullptr,
		&input_tensors[0], &input_values[0], input_values.size(),
		&output_tensors[0], &output_values[0], 3, //3 is the number of outputs count..
		nullptr, 0, nullptr, status
	);	if (TF_GetCode(status) != TF_OK)
	{
		printf("ERROR: SessionRun: %s", TF_Message(status));
	}
	auto detection_classes = static_cast<float_t*>(TF_TensorData(output_values[1]));
	auto detection_scores = static_cast<float_t*>(TF_TensorData(output_values[2]));
	auto detection_boxes = static_cast<float_t*>(TF_TensorData(output_values[0]));

	for (int i = 0; i < 6; i++) { //6 is not max number of detections of the network.
		std::cout << "Class Id: "<<detection_classes[i] <<" ";
		std::cout << "Confidence: " << detection_scores[i] << " ";
		for (int j = 0; j < 4; j++) {
			std::cout << "Cordinates: " << detection_boxes[1 * i + j] << " ";
		}
		std::cout << "\n";
	}
	std::cout << detection_boxes;

	cv::Rect rect = cv::Rect(detection_boxes[1] * image.cols, detection_boxes[0] * image.rows, (detection_boxes[3] - detection_boxes[1]) * image.cols, (detection_boxes[2] - detection_boxes[0]) * image.rows);
	rectangle(image, rect, cv::Scalar(0, 255, 0), 2, 8, 0);
	cv::imshow("test", image);
	cv::waitKey();

	
	//free memory
	TF_DeleteStatus(status);

	for (auto& t : output_values) {
		TF_DeleteTensor(t);
	}
	output_values.clear();
	output_values.shrink_to_fit();

	
	for (auto& t : input_values) {
		TF_DeleteTensor(t);
	}

	input_values.clear();
	input_values.shrink_to_fit();

	input_tensors.clear();
	input_tensors.shrink_to_fit();

}

void Network::Deallocator(void* data, size_t length, void* arg)
{
	printf("Dellocation of the input tensor");
	//std::free(data);
	//data = nullptr;
}

Network::~Network()
{
	TF_DeleteGraph(graph);
	TF_Status* status = TF_NewStatus();
	TF_CloseSession(session, status);
	if (TF_GetCode(status) != TF_OK) {
		printf("Error close session");
	}
	TF_DeleteSession(session, status);
	TF_DeleteStatus(status);
}



int main()
{
	std::string modelPath = "D:/Projects/Tensorflow_C_Api_Ssd/tf_2x/models/centernet_resnet50_v2_512x512_coco17_tpu-8_v2/saved_model";
	std::string imagePath = "D:/Projects/Tensorflow_C_Api_Ssd/tf_2x/20201007214411187_4239.png";
	
	Network ssd_network;
	ssd_network.LoadGraph(modelPath);

	cv::Mat image;
	image = cv::imread(imagePath);
	image.convertTo(image, CV_8UC3);
	//std::cout << image.channels;
	for (size_t i = 0; i < 100; i++) {
		ssd_network.Run(image);
	}
	ssd_network.~Network();
	image = NULL;
}