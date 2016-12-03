from pybrain.datasets.supervised import SupervisedDataSet;
from pybrain.tools.shortcuts import buildNetwork;
from pybrain.supervised.trainers.backprop import BackpropTrainer;
from matplotlib import pyplot as plt;
import timeit;
import logging;
from logging.config import fileConfig;

fileConfig('logging_config.ini');
logger = logging.getLogger();

class NeuralNetwork:
	""" Neural network class """
	def __init__(self, num_inputs, num_outputs, num_hidden_layers):
		self.num_inputs = num_inputs;
		self.num_outputs = num_outputs;
		self.num_hidden_layers = num_hidden_layers;
		self.dataset = None;
		self.network = None;
		self.backprop = None;
		self.epochs = 10;
		self.training_time = 0;
		logger.info("New neural network object created.");

	def run(self, operator):
		logger.info("Starting up neural network for %s.", operator);
		self.buildDataset();
		self.buildNetwork();

		# XOR data
		if operator is "XOR":
			self.addData([0,0], [0]);
			self.addData([0,1], [1]);
			self.addData([1,0], [1]);
			self.addData([1,1], [0]);

		# NXOR data
		if operator is "NXOR":
			self.addData([0,0], [1]);
			self.addData([0,1], [0]);
			self.addData([1,0], [0]);
			self.addData([1,1], [1]);

		self.backProp();
		self.testData();
		self.trainData();
		self.testData();
		self.graphResults();
		return True;

	def buildDataset(self):
		self.dataset = SupervisedDataSet(self.num_inputs, self.num_outputs);
		logger.info("Dataset object built.");
		return True;

	def addData(self, data_in, data_out):
		self.dataset.addSample(data_in, data_out);
		logger.info("Data appended.");
		return True;

	def buildNetwork(self):
		self.network = buildNetwork(self.num_inputs, self.num_hidden_layers, self.num_outputs);
		logger.info("Network created.");
		return True;

	def backProp(self):
		self.backprop = BackpropTrainer(self.network, learningrate = 0.01, momentum = 0.99);
		logger.info("Back propogation trainer created.");
		return True;

	def testData(self):
		logger.info('MSE: %s', str(self.backprop.testOnData(self.dataset)));
		return True;

	def trainData(self):
		start = timeit.default_timer();
		self.backprop.trainOnDataset(self.dataset, self.epochs);
		stop = timeit.default_timer();
		self.training_time = stop - start;
		logger.info("Training time: %s", str(self.training_time));
		return True;

	def graphResults(self):
		plt.figure(str(self.num_hidden_layers)+" hidden layers");
		plt.show();
		return True;

