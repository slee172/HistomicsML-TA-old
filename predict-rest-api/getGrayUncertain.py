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
    cdef long i, row, col, curX, curY
    cdef double scores, uncertainty
    cdef double SREGION_GRID_SIZE = 80.0
    cdef double KERN_SIZE = 7 * 11
    cdef double uncertMin, uncertMax
    cdef double classMin, classMax

    with nogil:
        for i in range(nrows):
            scores = (slideScores[i] * 2 ) - 1
            scoreVec[i] = 1 - fabs(scores)

    sort_cpp(scoreVec)

    cdef double uncertMedian = scoreVec[nrows / 2]

    with nogil:
        for i in range(nrows):
            curX = int(ceil(centX[i, 0] / SREGION_GRID_SIZE) - 1)
            curY = int(ceil(centY[i, 0] / SREGION_GRID_SIZE) - 1)
            # uncertainty = 1 - fabs(slideScores[i, 0])
            scores = (slideScores[i] * 2 ) - 1
            uncertainty = 1 - fabs(scores)

            if uncertainty > uncertMedian:
                uncertainMap[curX, curY] += 1.0

            if scores >= 0:
                classMap[curX, curY] += 1.0

            densityMap[curX, curY] += 1.0

    with nogil:
        for row in range(fX):
            for col in range(fY):
                if densityMap[row, col] == 0:
                    classMap[row, col] = 0
                    uncertainMap[row, col] = 0
                else:
                    classMap[row, col] = classMap[row, col] / densityMap[row, col]
                    uncertainMap[row, col] = uncertainMap[row, col] / densityMap[row, col]

    uncertainMap = filters.gaussian(uncertainMap, 1)
    classMap = filters.gaussian(classMap, 1)

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

    with nogil:
        for row in range(fX):
            for col in range(fY):
                grayUncertain[row, col] = int(min(255.0 * uncertainMap[row, col]/ uncertMax, 255.0))
                grayClass[row, col] = int(min(255.0 * classMap[row, col] / classMax, 255.0))


    return np.asarray(grayUncertain), np.asarray(grayClass), uncertMin, uncertMax, uncertMedian, classMin, classMax
