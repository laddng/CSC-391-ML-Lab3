from classes.NeuralNetwork import *;

def main():
	""" Run neural network """

	# 2 hidden layers
	net = NeuralNetwork(2, 1, 4);
	net.run();

	# 200 hidden layers
	net = NeuralNetwork(2, 1, 200);
	net.run();

	return True;

main();

