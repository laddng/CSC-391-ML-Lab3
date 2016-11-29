from matplotlib import pyplot as plt;

def graphResults(RMSE_array):
	""" Prints out the RMSE for each iteration of our neural network """
	num_iterations = len(RMSE_array);

	for i in num_iterations:
		print(str(RMSE_array[i]));

	return True;

