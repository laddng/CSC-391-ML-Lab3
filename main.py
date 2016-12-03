from classes.NeuralNetwork import *;
from copy import deepcopy, copy;

def main():
	""" Run neural network tests """
	# 10 different hidden layers
	for i in range(10):
		# 10 different epochs
		epochs = [];
		RMSE = [];
		for k in range(10):
			hidden_layers = (i+1)*10;
			net = NeuralNetwork(2, 1, hidden_layers);
			net.run("NXOR", (10*k));
			epochs.append(deepcopy(net.epochs));
			RMSE.append(deepcopy(net.RMSE));
		graphResults(epochs, RMSE, ("# epochs, "+str(hidden_layers)+" hidden layers"), "RMSE");

	# 10 different hidden layers
	time = [];
	layers = [];
	for i in range(10):
		hidden_layers = (i+1)*10;
		net = NeuralNetwork(2, 1, hidden_layers);
		# 1000 different epochs
		net.run("XOR", 1000);
		time.append(deepcopy(net.training_time));
		layers.append(deepcopy(hidden_layers));
	graphResults(layers, time, "# hidden layers", "time");

	return True;

def graphResults(x,y,xlabel,ylabel):
	""" Given x and y, plots the results
	to a matplotlib graph for output """
	plt.figure("Results");
	plt.xlabel(xlabel);
	plt.ylabel(ylabel);
	plt.plot(x,y);
	plt.show();

main();

