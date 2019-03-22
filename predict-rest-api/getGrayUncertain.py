import numpy as np
from skimage import filters
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libc.math cimport fabs, ceil

def sort_cpp(double[::1] a):
    sort(&a[0], (&a[0]) + a.shape[0])

def get(long fX, long fY, double[:, :] centX, double[:, :] centY, double[:] slideScores not None):

    cdef long nrows = slideScores.shape[0]
    cdef double[:, :] uncertainMap = np.zeros((fX, fY), dtype=np.double)
    cdef double[:, :] classMap = np.zeros((fX, fY), dtype=np.double)
    cdef double[:, :] densityMap = np.zeros((fX, fY), dtype=np.double)
    cdef long[:, :] grayUncertain = np.zeros((fX, fY), dtype=np.int)
    cdef long[:, :] grayClass = np.zeros((fX, fY), dtype=np.int)
    cdef double[:] scoreVec = np.ones((nrows,), dtype=np.double)
    cdef long i, row, col, curX, curY, uncertAbove, uncertBelow
    cdef double scores, uncertainty
    cdef double SREGION_GRID_SIZE = 80.0
#     cdef double KERN_SIZE = 7 * 11
    cdef double uncertMin, uncertMax
    cdef double classMin, classMax

    with nogil:
        for i in range(nrows):
            curX = int(ceil(centX[i, 0] / SREGION_GRID_SIZE) - 1)
            curY = int(ceil(centY[i, 0] / SREGION_GRID_SIZE) - 1)
            scores = (slideScores[i] * 2 ) - 1
            scoreVec[i] = 1 - fabs(scores)

            if uncertainty > 0.5:
                uncertAbove += 1
            else:
                uncertBelow += 1


            uncertainMap[curX, curY] = max(uncertainMap[curX, curY], scoreVec[i])

            if scores >= 0:
                classMap[curX, curY] += 1.0

            densityMap[curX, curY] += 1.0

#     sort_cpp(scoreVec)
#     cdef double uncertMedian = scoreVec[nrows / 2]

    with nogil:
        for row in range(fX):
            for col in range(fY):
                if densityMap[row, col] == 0:
                    classMap[row, col] = 0
#                     uncertainMap[row, col] = 0
                else:
                    classMap[row, col] = classMap[row, col] / densityMap[row, col]
#                     uncertainMap[row, col] = uncertainMap[row, col] / densityMap[row, col]

    uncertainMap = filters.gaussian(uncertainMap, 4)
    classMap = filters.gaussian(classMap, 4)


    uncertMax = uncertainMap[0, 0]
    with nogil:
       for i in range(fX):
           for j in range(fY):
               if uncertainMap[i, j] > uncertMax:
                   uncertMax = uncertainMap[i, j]
               if uncertainMap[i, j] < uncertMin:
                   uncertMin = uncertainMap[i, j]

    classMax = classMap[0, 0]
    with nogil:
       for i in range(fX):
           for j in range(fY):
               if classMap[i, j] > classMax:
                   classMax = classMap[i, j]
               if classMap[i, j] < classMin:
                   classMin = classMap[i, j]



#     cdef long HIST_BINS=20
#     cdef long[:] uncertHist = np.zeros((HIST_BINS,), dtype=np.int)
#     cdef long index, total
#     cdef double uncertNorm, uncertrange
#     cdef double UNCERT_PERCENTILE = 0.9
    cdef double uncertPercent = uncertAbove / (fY * fX)
#     uncertrange = uncertMax - uncertMin

#     with nogil:
#        for i in range(fX):
#            for j in range(fY):
#             index = int(min(uncertainMap[i, j] / uncertrange * HIST_BINS, (HIST_BINS - 1)))
#             uncertHist[index] = uncertHist[index] + 1

#     total = 0
#     with nogil:
#        for i in range(HIST_BINS):
#             total = total + uncertHist[i]
#             if total > uncertPercent:
#                 index = i
#                 break

#     uncertNorm = index / float(HIST_BINS)



    with nogil:
       for row in range(fX):
           for col in range(fY):
               grayUncertain[row, col] = int(min(255.0 * uncertainMap[row, col]/ uncertMax, 255.0))
               grayClass[row, col] = int(min(255.0 * classMap[row, col] / classMax, 255.0))


    return np.asarray(grayUncertain), np.asarray(grayClass), uncertMin, uncertMax, uncertPercent, classMin, classMax
