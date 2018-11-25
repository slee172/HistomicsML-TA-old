import large_image
import mysql.connector
import numpy as np
import cv2
from scipy.misc import imsave


inputImageFile ="/home/sanghoon/docker-py/hmlWeb/TCGA-3C-AALJ-01Z-00-DX1.svs.dzi.tif"
slideName = 'TCGA-3C-AALJ-01Z-00-DX1'

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

imsave('./test.png', im_out)
