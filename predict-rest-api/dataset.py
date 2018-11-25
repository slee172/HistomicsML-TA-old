import h5py
import numpy as np

import settings


class Dataset():

	def __init__(self, path):

		self.f = h5py.File(path)
		self.features = self.f['features'][:]
		self.x_centroid = self.f['x_centroid'][:]
		self.y_centroid = self.f['y_centroid'][:]
		self.slideIdx = self.f['slideIdx'][:]
		self.slides = self.f['slides'][:]
		self.dataIdx = self.f['dataIdx'][:]
		# self.length = self.f['dataIdx'][:]
		self.n_slides = len(self.dataIdx)
		self.n_objects = len(self.slideIdx)

		s = settings.Settings()
		self.FEATURE_DIM = s.FEATURE_DIM

	def getSlideIdx(self, slide):
		idx = np.argwhere(self.slides == slide)[0, 0]
		return idx

	def getDataIdx(self, index):
		idx = self.dataIdx[index][0]
		return idx

	def getObjNum(self, index):
		if self.n_slides > index + 1:
			num = self.dataIdx[index + 1, 0] - self.dataIdx[index, 0]
		else:
			num = self.n_objects - self.dataIdx[index, 0]
		return num

	def getFeatureSet(self, index, num):
		fset = self.features[index: index+num]
		return fset

	def getXcentroidSet(self, index, num):
		xset = self.x_centroid[index: index+num]
		return xset

	def getYcentroidSet(self, index, num):
		yset = self.y_centroid[index: index+num]
		return yset

	def getSlideIdxSet(self, index, num):
		idxset = self.slideIdx[index: index+num]
		return idxset
