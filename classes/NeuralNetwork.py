import numpy as np;
from modules.graph_utils import *;
from classes.HiddenLayer import *;

class NeuralNetwork:
	""" A class for our neural network """
	def __init__(self, inputs, num_hidden_layers):
		self.num_hidden_layers = num_hidden_layers;
		self.inputs = inputs;
		print("[NeuralNetwork]: New neural network object created.");

	def createHiddenLayers(self, num_hidden_layers);
		print("[NeuralNetwork]: Creating "+str(num_hidden_layers)+" hidden layers...");
		for i in range(num_hidden_layers):
			hidden_layer = HiddenLayer();
			self.hidden_layers.append(hidden_layer);
		print("[NeuralNetwork]: Finished creating "+str(num_hidden_layers)+" hidden layers...");

		return True;

	def forwardPropogate(self):
		return True;

	def backwardPropogate(self):
		return True;

