
#include "pch.h"
#include <iostream>
#include "tensorflow/c/c_api.h"
#include <codecvt>
#include <fstream>
#include <vector>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

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
	TF_Buffer* buffer = ReadBufferFromFile(modelPath);
	if (buffer == nullptr) {
		throw std::invalid_argument("Error creating the session from the given model path %s !");
	}
	TF_Status* status = TF_NewStatus();
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

	graph = TF_NewGraph();
	TF_GraphImportGraphDef(graph, buffer, opts, status);
	TF_DeleteImportGraphDefOptions(opts);
	TF_DeleteBuffer(buffer);
	if (TF_GetCode(status) != TF_OK) {
		TF_DeleteGraph(graph);
		graph = nullptr;
	}
	TF_DeleteStatus(status);

	// create session from graph
	status = TF_NewStatus();
	TF_SessionOptions* options = TF_NewSessionOptions();
	session = TF_NewSession(graph, options, status);
	TF_DeleteSessionOptions(options);
}

void Network::Run(cv::Mat image)
{
	std::vector<TF_Output> 	input_tensors, output_tensors;
	std::vector<TF_Tensor*> input_values, output_values;

	//input tensor shape.
	int num_dims = 4;
	std::int64_t input_dims[4] = {1, image.rows, image.cols, 3}; //1 is number of batch, and 3 is the no of channels.
	int num_bytes_in = image.cols * image.rows * 3; //3 is the number of channels.
	
	input_tensors.push_back({TF_GraphOperationByName(graph, "image_tensor"),0});
	input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));

	output_tensors.push_back({ TF_GraphOperationByName(graph, "detection_classes"),0 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "detection_scores"),0 });
	output_values.push_back(nullptr);

	output_tensors.push_back({ TF_GraphOperationByName(graph, "detection_boxes"),0 });
	output_values.push_back(nullptr);

	TF_Status* status = TF_NewStatus();
	TF_SessionRun(session, nullptr,
		&input_tensors[0], &input_values[0], input_values.size(),
		&output_tensors[0], &output_values[0], 3, //3 is the number of outputs count..
		nullptr, 0, nullptr, status
	);
	if (TF_GetCode(status) != TF_OK)
	{
		printf("ERROR: SessionRun: %s", TF_Message(status));
	}
	
	auto detection_classes = static_cast<float_t*>(TF_TensorData(output_values[0]));
	auto detection_scores = static_cast<float_t*>(TF_TensorData(output_values[1]));
	auto detection_boxes = static_cast<float_t*>(TF_TensorData(output_values[2]));


	for (int i = 0; i < 6; i++) { //6 is not max number of detections of the network.
		std::cout << "Class Id: "<<detection_classes[i] <<" ";
		std::cout << "Confidence: " << detection_scores[i] << " ";
		for (int j = 0; j < 4; j++) {
			std::cout << "Cordinates: " << detection_boxes[i] << " ";
		}
		std::cout << "\n";
	}
	std::cout << detection_classes;
	//free output
	TF_DeleteStatus(status);

	inputs.clear();
	inputs.shrink_to_fit();

	for (auto& t : input_values) {
		TF_DeleteTensor(t);
	}

	input_values.clear();
	input_values.shrink_to_fit();

	/*for (size_t i = 0; i < output_values.size(); ++i)
	{
		const auto data = static_cast<float*>(TF_TensorData(output_values.at(i)));
		printf("%f", data[1]);
	}*/
	for (auto& t : output_values) {
		TF_DeleteTensor(t);
	}
	output_values.clear();
	output_values.shrink_to_fit();
}

void Network::Deallocator(void* data, size_t length, void* arg)
{
	TF_DeleteGraph(graph);
	status = TF_NewStatus();
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(SessionOpts);
	TF_DeleteStatus(status);
	std::free(data);
}

int main()
{
	std::string modelPath = "D:/Projects/1_HAZMAT/trunk/PVRE_DE_SICK_HAZMAT/bin/PVRE/ProtobufFiles/HAZMAT_detector_binary_20180914.pb";
	std::string imagePath = "D:/Images/Hazmat/SICK/2018_07_19_01_Single_Labels/01_Single_Labels/12.x/12.1/0d886616ccfbc4911d36b7ca731dcb1c778a22d0.jpg";
	
	Network ssd_network;
	ssd_network.LoadGraph(modelPath);

	cv::Mat image;
	image = cv::imread(imagePath);
	//cv::imshow("test", image);
	//cv::waitKey();
	image.convertTo(image, CV_8UC3);
	//std::cout << image.channels;
	ssd_network.Run(image);
}
