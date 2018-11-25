import numpy as np


class label():
	def __init__(self):

		self.uid = None
		self.slide = None
		self.trainSet = None
		self.dataSet = None
		self.left = 0
		self.top = 0
		self.width = 0
		self.height = 0
		self.inFile = None
		self.outFile = None

	def setData(self, q):
		self.uid = q["uid"]
		self.trainSet = '/localdata/classifiers/' + str(q["trainset"]) + '.h5'
		self.dataSet = str(q["dataset"])
		self.slide = str(q["slide"])
		self.left = int(q["left"])
		self.top = int(q["top"])
		self.width = int(q["width"])
		self.height = int(q["height"])
		self.bottom = self.top + self.height
		self.right = self.left + self.width
		self.inFile = '/localdata/classifiers/tmp/' + self.slide + '_' + str(self.left) + \
		'_' + str(self.top) + '_' + str(self.width) + '_' + str(self.height) + '_' + '_.jpg'
		self.outFile = 'trainingsets/tmp/' + self.slide + '_' + str(self.left) + \
		'_' + str(self.top) + '_' + str(self.width) + '_' + str(self.height) + '_' + '_.jpg'
