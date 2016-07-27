
import numpy as np

def HDIofGrid(probMassVec, credMass=0.95):
    # Arguments:
    #   probMassVec is a vector of probability masses at each grid point.
    #   credMass is the desired mass of the HDI region.
    # Return value:
    #   A dict with keys:
    #   indices is a vector of indices that are in the HDI
    #   mass is the total mass of the included indices
    #   height is the smallest component probability mass in the HDI

    sortedProbMass = np.sort(probMassVec)[::-1]

    HDIheightIdx = np.nonzero(np.cumsum(sortedProbMass) >= credMass)[0]
    HDIheightIdx = np.min(HDIheightIdx)

    HDIheight = sortedProbMass[HDIheightIdx]

    indices = np.nonzero(probMassVec >= HDIheight)[0]
    HDImass = sum(probMassVec[indices])

    return {
        'indices': indices,
        'mass': HDImass,
        'height': HDIheight
    }
