from classes.NeuralNetwork import *;

def main():
	""" Run neural network tests """

	# 2 hidden layers
	net = NeuralNetwork(2, 1, 4);
	net.run("NXOR");

	# 200 hidden layers
	net = NeuralNetwork(2, 1, 200);
	net.run("XOR");

	return True;

main();

