// NeuralNetwork.hpp 
#include "/home/root/ddpg_actor/eigen/Eigen/Dense"
#include <iostream> 
#include <vector> 

// use typedefs for future ease for changing data types like : float to double 
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;
typedef  int state_dim;
typedef  int fc1 ;
typedef  int fc2 ;
typedef  int fc3 ;
typedef  int fc4 ;
typedef  int fc4_w;
typedef  int fc4_n;
typedef  int fc4_p;
 //neural network implementation class! 
class NeuralNetwork {
public:
	NeuralNetwork( state_dim sd= state_dim(10),fc1 l1= fc1(400), fc2 l2=fc2(300), fc3 l3=fc3(200),fc4 l4= fc4(100),fc4_w w=fc4_w(1),fc4_n n=fc4_n(1), fc4_p p=fc4_p(1), float* w_app=0, float* n_app=0, float* p_app=0 );
	Matrix ReLu(Matrix x);
	float Sigmoid(float app);
	Matrix readCSV(std::string filename, Matrix a);
	Matrix readCSV_bias(std::string filename, Matrix a);
	state_dim sd;
	fc1 l1;
	fc2 l2;
	fc3 l3;
	fc4 l4;
	fc4_w w;
	fc4_n n;
	fc4_p p;
}; 
