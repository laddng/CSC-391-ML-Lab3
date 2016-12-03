from classes.NeuralNetwork import *;

def run():
	""" Run neural network """

	# 2 hidden layers
	net = NeuralNetwork(2, 1, 4);
	net.buildDataset();
	net.buildNetwork();
	net.addData([0,0], [0]);
	net.addData([0,1], [0]);
	net.addData([1,0], [0]);
	net.addData([1,1], [1]);
	net.backProp();
	net.testData();
	net.trainData();
	net.testData();

	# 200 hidden layers
	NeuralNetwork(2, 1, 200);

	return True;

run();

