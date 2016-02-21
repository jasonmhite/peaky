from numpy import *
import pandas as pd
from matplotlib.pyplot import *
from copy import deepcopy

__all__ = ["Spectrum"]

class Spectrum(object):
    def __init__(self, live, real, ind, counts, label=None):
        self.live = live
        self.real = real
        self.ind = ind
        self.counts = counts
        self.label = label

        self._s = pd.Series(
            data=self.counts,
            index=self.ind,
        )

    def __sub__(self, other):
        return ChannelSpectrum(
            self.live,
            self.real,
            self.ind,
            self.counts - (self.live / other.live) * other.counts
        )

    def array(self):
        return self.vstack([self.channels, self.counts]).T

    def series(self):
        return self._s

    @classmethod
    def load(cls, file, live, real, label=None):
        data = genfromtxt(file)
        return cls(live, real, data[:, 0], data[:, 1], label)

    def tr(self, f):
        # Apply translation f to index
        newS = deepcopy(self)
        newS.ind = f(newS.ind)

        return newS

    def __getitem__(self, k):
        ind = (self.ind >= k[0]) & (self.ind <= k[1])

        return vstack((self.ind[ind], self.counts[ind])).T

    def plot(self, name=None, scale=1.0, logv=True, alpha=1.0):
        if name is None:
            name = self.label

        if logv:
            semilogy(self.ind, scale * self.counts, label=name, alpha=alpha)
        else:
            plot(self.ind, scale * self.counts, label=name, alpha=alpha)
