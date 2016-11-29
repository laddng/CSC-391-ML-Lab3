import os;

def importData(folder):
	""" Imports data from a given folder """
	print("[importData]: Importing data...");
	data = [];

	for data_file in os.lisdir(folder):
		print(data_file);
		data.append(data);
	print("[importData]: Finished importing data.");

	return data;

