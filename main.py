from modules.data_utils import *;
from classes.NeuralNetwork import *;

def run():
	""" Runs our neural network program """
	data = importData("data/");

	# 2 hidden layer
	results = NeuralNetwork(data, 2);

	# 200 hidden layers
	results = NeuralNetwork(data, 200);

	return True;

run();

