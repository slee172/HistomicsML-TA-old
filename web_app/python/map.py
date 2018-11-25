import sys,os
import numpy as np
import h5py
from keras.utils.np_utils import to_categorical
from time import time
import tensorflow as tf
import networks
import json
import large_image
import mysql.connector
import cv2
from scipy.misc import imsave

def create_label_img(slideName):

    inputImageFile = '/localdata/pyramids/BRCA/' + slideName + '.svs.dzi.tif'

    left = 50000
    top = 35000
    width = 2000
    height = 2000
    bottom = top + height
    right = left + width

    bold = 512
    bold_left = left - bold
    bold_top = top - bold
    bold_bottom = bottom + bold
    bold_right = right + bold
    bold_width = width + 2*bold
    bold_height = height + 2*bold

    ts = large_image.getTileSource(inputImageFile)

    region = dict(
        left=left, top=top,
        width=width, height=height,
    )

    im_region = ts.getRegion(
        region=region, format=large_image.tilesource.TILE_FORMAT_NUMPY
    )[0]

    mydb = mysql.connector.connect(
      host="localhost",
      user="guest",
      passwd="guest",
      database="nuclei",
      charset='utf8',
      use_unicode=True
    )

    boundaryTablename = 'sregionboundaries'

    runcursor = mydb.cursor()

    query = 'SELECT boundary from ' + boundaryTablename + ' where slide="' +  slideName + \
    '" AND centroid_x BETWEEN ' + str(left) + ' AND ' + str(right) + \
    ' AND centroid_y BETWEEN ' + str(top) + ' AND ' + str(bottom)

    runcursor.execute(query)

    boundarySet = runcursor.fetchall()

    # set an array for boundary points in a region to zero
    # boundaryPoints = np.zeros((1, 2), dtype=np.int32)
    boundaryPoints = []
    # b_index = 0
    for b in boundarySet:
      object = b[0].encode('utf-8').split(' ')
      object_points = []
      for p in range(len(object)-1):
          intP = map(int, object[p].split(','))
          intP[0] = intP[0] - left + bold
          intP[1] = intP[1] - top + bold
          object_points.append(intP)
      boundaryPoints.append(np.asarray(object_points))

    im_bold = np.zeros((bold_width, bold_height), dtype=np.uint8)

    cv2.fillPoly(im_bold, boundaryPoints, 255)

    im_out = im_bold[bold:bold+width, bold:bold+width]

    return im_out


if len (sys.argv) != 4 :
    print ('Usage: python conut.py trainSet dataSet outFile')
    sys.exit (1)

trainSet = sys.argv[1]
dataSet = sys.argv[2]
slideName = sys.argv[3]
outFile = sys.argv[4]

# load dataset
f = h5py.File(dataSet)
features = f['features'][:]
# load trainingset
c = h5py.File(trainSet)
object_num = len(c['slideIdx'][:])
sample_features = c['features'][:]
train_features = np.vstack((sample_features, c['augments_features'][:]))

sample_labels = c['labels'][:]
train_labels = np.vstack((sample_labels, c['augments_labels'][:]))
train_labels[train_labels<0] = 0
train_labels = to_categorical(train_labels, num_classes=2)

# initialize neural network model
model = networks.Network()
model.init_model()

print('Training ... ', len(train_labels))
t0 = time()
model.train_model(train_features, train_labels)
t1 = time()
print('Training took ', t1 - t0)

print('Predict ... ', len(train_labels))
t0 = time()
predicts = model.predict(features)
t1 = time()
print('Predict took ', t1 - t0)
