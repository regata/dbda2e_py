
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

    credMass = credMass * sum(probMassVec)

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

def plotPost(trace, ax, title, hdi=0.95, xlabel=r'$\theta$'):

    def annotate(theta, prob, ax, hdi=0.95):
        mode_text = 'mode = %.2f' % theta[np.argmax(prob)]
        ax.annotate(mode_text, xy=(0.98, 0.9), xycoords='axes fraction', fontsize=12,
                    horizontalalignment='right')

        if not hdi:
            return

        # draw HDI
        HDIinfo = HDIofGrid(prob , credMass=hdi)
        hdi_x = theta[HDIinfo['indices']]
        hdi_y = np.full_like(hdi_x, HDIinfo['height'])
        ax.plot(hdi_x, hdi_y, marker='.', color='k', ls='')

        ax.annotate('%.2f' % hdi_x[0], xy=(hdi_x[0], hdi_y[0]*1.1),
                    horizontalalignment='right', verticalalignment='bottom', fontsize=12)
        ax.annotate('%.2f' % hdi_x[-1], xy=(hdi_x[-1], hdi_y[-1]*1.1),
                    horizontalalignment='left', verticalalignment='bottom', fontsize=12)

        hdi_text = '%.0f%% HDI' % (hdi * 100)
        hdi_mid_idx = len(hdi_x) // 2 
        ax.annotate(hdi_text, xy=(hdi_x[hdi_mid_idx], 1.3*hdi_y[hdi_mid_idx]),
                    horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    heights, bins, patches = ax.hist(trace, bins=50, color='cornflowerblue', normed=True)
    bins = (bins + (bins[1]-bins[0]))[:-1] # convert to bin centers from bin edges
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    annotate(bins, heights, ax, hdi)
