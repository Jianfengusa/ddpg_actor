#include <iostream>
#include "/home/root/ddpg_actor/eigen/Eigen/Dense"
#include <vector> 
#include <fstream>
#include <string.h>
#include "/home/root/ddpg_actor/NeuralNetwork.hpp" 
using namespace std;
NeuralNetwork::NeuralNetwork(state_dim sd, fc1 l1, fc2 l2, fc3 l3, fc4 l4, fc4_w w, fc4_n n, fc4_p p,float* w_app,float *n_app,float *p_app) {
	Matrix input = Matrix(1, sd);
	Matrix first_layer = Matrix(sd, l1);
	Matrix second_layer = Matrix(l1, l2);
	Matrix third_layer = Matrix(l2, l3);
	Matrix fourth_layer = Matrix(l3, l4);
	Matrix w_layer = Matrix(l4, w);
	Matrix n_layer = Matrix(l4, n);
	Matrix p_layer = Matrix(l4, p);

	Matrix first_layer_bias = Matrix(1, l1);
	Matrix second_layer_bias = Matrix(1, l2);
	Matrix third_layer_bias = Matrix(1, l3);
	Matrix fourth_layer_bias = Matrix(1, l4);
	Matrix w_layer_bias = Matrix(1, w);
	Matrix n_layer_bias = Matrix(1, n);
	Matrix p_layer_bias = Matrix(1, p);
	Matrix x = Matrix();

	/*string filename_input= "E:/Github/Irrigation_Project/GPS/sensor_output.csv";
	string filename_l1_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc1_weight.csv";
	string filename_l2_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc2_weight.csv";
	string filename_l3_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc3_weight.csv";
	string filename_l4_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_weight.csv";
	string filename_w_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_w_weight.csv";
	string filename_n_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_n_weight.csv";
	string filename_p_weight = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_p_weight.csv";

	string filename_l1_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc1_bias.csv";
	string filename_l2_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc2_bias.csv";
	string filename_l3_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc3_bias.csv";
	string filename_l4_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_bias.csv";
	string filename_w_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_w_bias.csv";
	string filename_n_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_n_bias.csv";
	string filename_p_bias = "E:/Github/Irrigation_Project/DDPG_FER_2/fc4_p_bias.csv";*/

	string filename_input = "/home/root/ddpg_actor/input.csv";
	string filename_l1_weight = "/home/root/ddpg_actor/fc1_weight.csv";
	string filename_l2_weight = "/home/root/ddpg_actor/fc2_weight.csv";
	string filename_l3_weight = "/home/root/ddpg_actor/fc3_weight.csv";
	string filename_l4_weight = "/home/root/ddpg_actor/fc4_weight.csv";
	string filename_w_weight = "/home/root/ddpg_actor/fc4_w_weight.csv";
	string filename_n_weight = "/home/root/ddpg_actor/fc4_n_weight.csv";
	string filename_p_weight = "/home/root/ddpg_actor/fc4_p_weight.csv";

	string filename_l1_bias = "/home/root/ddpg_actor/fc1_bias.csv";
	string filename_l2_bias = "/home/root/ddpg_actor/fc2_bias.csv";
	string filename_l3_bias = "/home/root/ddpg_actor/fc3_bias.csv";
	string filename_l4_bias = "/home/root/ddpg_actor/fc4_bias.csv";
	string filename_w_bias = "/home/root/ddpg_actor/fc4_w_bias.csv";
	string filename_n_bias = "/home/root/ddpg_actor/fc4_n_bias.csv";
	string filename_p_bias = "/home/root/ddpg_actor/fc4_p_bias.csv";


	
	first_layer = readCSV(filename_l1_weight, first_layer);
	second_layer = readCSV(filename_l2_weight, second_layer);
	third_layer = readCSV(filename_l3_weight, third_layer);
	fourth_layer = readCSV(filename_l4_weight, fourth_layer);
	w_layer = readCSV(filename_w_weight, w_layer);
	n_layer = readCSV(filename_n_weight, n_layer);
	p_layer = readCSV(filename_p_weight, p_layer);
	
	input = readCSV(filename_input, input);
	first_layer_bias = readCSV_bias(filename_l1_bias, first_layer_bias);
	second_layer_bias = readCSV_bias(filename_l2_bias, second_layer_bias);
	third_layer_bias = readCSV_bias(filename_l3_bias, third_layer_bias);
	fourth_layer_bias = readCSV_bias(filename_l4_bias, fourth_layer_bias);
	w_layer_bias = readCSV_bias(filename_w_bias, w_layer_bias);
	n_layer_bias = readCSV_bias(filename_n_bias, n_layer_bias);
	p_layer_bias = readCSV_bias(filename_p_bias, p_layer_bias);
	
	//cout << first_layer_bias << endl;
	
	x = input * first_layer + first_layer_bias;
	x = ReLu(x);
	x = x * second_layer + second_layer_bias;
	x = ReLu(x);
	x = x * third_layer + third_layer_bias;
	x = ReLu(x);
	x =	x * fourth_layer + fourth_layer_bias;
	x = ReLu(x);
	*w_app = (x * w_layer + w_layer_bias)(0, 0);
	*w_app = Sigmoid(*w_app)*35;
	if (input(0, 0) < 86)
	{
		*n_app = (x * n_layer + n_layer_bias)(0, 0);
		*n_app = Sigmoid(*n_app) * 35;
	}
	else
	{
		*n_app = 0.0;
	}
	if (input(0.0) < 96)
	{
		*p_app = (x * p_layer + p_layer_bias)(0, 0);
		*p_app = Sigmoid(*p_app) * 35;
	}
	else
	{
		*p_app = 0.0;
	}
}
Matrix NeuralNetwork::ReLu(Matrix x)
{
	for (int i = 0, row = x.rows(); i < row; i++ ) {
		for (int j = 0, col = x.cols(); j < col; j++) {
			if (x(i, j) < 0) {
				x(i, j) = 0;
			}
			else {}
		}
	}
	return x;
}
float NeuralNetwork::Sigmoid(float app)
{
	return exp(app) / (exp(app) + 1);
}
Matrix NeuralNetwork::readCSV(string filename, Matrix a)
{
	ifstream file(filename);
	string line;
	string num;
	vector<vector<float>> saved_model;
	if (!file.good())
	{
		cout << "Input file does not exist, please double check.  " << filename << endl;
		throw;
	}
	while (getline(file,line))
	{
		istringstream sin(line); 
		vector<float> fields;
		string field;
		while (getline(sin, field, ',')) 
		{
			fields.push_back(stof(field)); 
		}
		saved_model.push_back(fields);
	}
	if ((a.rows()!=saved_model.size()) || (a.cols()!=saved_model[0].size()))
	{
		cout << "Row or Col of the Input Matrix does not match the saved model dim" << endl;
		cout << "Row of input matrix:  " << a.rows() << "  Row of saved model:  " << saved_model.size() << endl;
		cout << "Col of input matrix:  " << a.cols() << "  Col of saved model:  " << saved_model[0].size() << endl;
		cout << filename << endl;
		throw;
	}
	for (int row = 0;row < saved_model.size();row++)
	{
		for (int col = 0;col < saved_model[row].size();col++)
		{
			a(row, col) = saved_model[row][col];
		}
	}
	return a;
}
Matrix NeuralNetwork::readCSV_bias(std::string filename, Matrix a)
{
	ifstream file(filename);
	string line;
	string num;
	vector<vector<float>> saved_model;
	if (!file.good())
	{
		cout << "Input file does not exist, please double check.  " << filename << endl;
		throw;
	}
	while (getline(file, line))
	{
		istringstream sin(line);
		vector<float> fields;
		string field;
		while (getline(sin, field, ','))
		{
			fields.push_back(stof(field));
		}
		saved_model.push_back(fields);
	}
	if ((a.rows() != saved_model[0].size()) || (a.cols() != saved_model.size()))
	{
		cout << "Row or Col of the Input Matrix does not match the saved model dim" << endl;
		cout << "Row of input matrix:  " << a.rows() << "  Row of saved model:  " << saved_model[0].size() << endl;
		cout << "Col of input matrix:  " << a.cols() << "  Col of saved model:  " << saved_model.size() << endl;
		cout << filename << endl;
		throw;
	}
	for (int row = 0;row < saved_model[0].size();row++)
	{
		for (int col = 0;col < saved_model.size();col++)
		{
			a(row, col) = saved_model[col][row];
		}
	}
	return a;
}
int main()
{	
	float w_app=10, n_app=10, p_app=10;
	float* w_ = &w_app, * n_ = &n_app, * p_ = &p_app;
	NeuralNetwork(10,40,30,20,10,1,1,1,w_,n_,p_);
	std::cout << w_app << std::endl;
	std::cout << n_app << std::endl;
	std::cout << p_app << std::endl;
	//std::cout << vec << std::endl;
	/*BoosterHandle handle;
	DMatrixHandle x;
	const char* xgb_path = "E:/Github/Irrigation_Project/Prediction_models_corn_fer/Saved_models/xgb_ni_0";
	int j=XGBoosterLoadModel(handle, xgb_path);
	cout << j << endl;*/
	return 0;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
