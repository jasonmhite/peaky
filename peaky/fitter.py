import scipy.optimize as sop

from numpy import *
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import quad
from matplotlib.pyplot import *

__all__ = ['PeakFitter']

ASCALE = norm.pdf(0)

def _check_fit(f):
    def wrapper(self, *args, **kwargs):
        if self.fit_params is None:
            self.fit()

        return f(self, *args, **kwargs)

    return wrapper

class PeakFitter(object):
    def __init__(self, ind, counts, l, u):
        self.ind = ind
        self.counts = counts
        self.l, self.u = l, u

        _ind = (self.ind >= self.l) & (self.ind <= self.u)
        self.peak_ind = self.ind[_ind]
        self.peak_counts = self.counts[_ind]

        self.fit_params = None
        self.calc_baseline()

    def calc_baseline(self):
        base_ind = (self.ind < self.l) | (self.ind > self.u)
        self.baseline_coeff = polyfit(self.ind[base_ind], self.counts[base_ind], deg=5)

        self.baseline = polyval(self.baseline_coeff, self.ind)

    def _gaussian(self, X, a, c, w):
        x_b = (X <= self.u) & (X >= self.l)
        R = zeros_like(X)

        R[x_b] = (a / ASCALE) * norm.pdf(X[x_b], loc=c, scale=(1. / w))

        return R

    def _skewgaussian(self, X, s, a, c, w):
        Y = w * (X - c)

        R = 2. * norm.pdf(Y) * norm.cdf(s * Y)
        c_adj = Y[R.argmax()]

        Cc = c - c_adj
        Yc = w * (X - Cc)

        V = 2. * norm.pdf(Yc) * norm.cdf(s * Yc)

        skval = (a / V.max()) * V

        return where(
            (X <= self.u) & (X >= self.l),
            skval,
            zeros_like(X),
        )

    def _eval_baseline(self, X):
        return polyval(self.baseline_coeff, X)

    def fit(self):
        # Remove baseline

        i_x = linspace(self.l, self.u, 100)
        peak_counts = self.peak_counts - self._eval_baseline(self.peak_ind)

        i_y = interp1d(
            self.peak_ind,
            peak_counts,
            kind="cubic",
            bounds_error=False,
            fill_value=0.0
        )(i_x)

        def fobj(C):
            return ((self._gaussian(i_x, *C) - i_y) ** 2).sum()

        cal = sop.differential_evolution(
            fobj,
            (
                (0.5 * peak_counts.max(), 1.5 * peak_counts.max()),
                (self.l, self.u),
                (1, self.u - self.l)
            ),
            tol=.0001,
            popsize=30,
        )

        if not cal.success:
            print("Fit failed")
        else:
            self.fit_params = cal.x

    def __call__(self, Xr, with_baseline=False):
        if self.fit_params is None:
            self.fit()

        X = atleast_1d(Xr)
        if with_baseline:
            res = self._eval_baseline(X)
        else:
            res = zeros_like(X, dtype=float64)

        res[(X >= self.l) & (X <= self.u)] += self._gaussian( X[(X >= self.l) & (X <= self.u)], *self.fit_params)
        # res[(X >= self.l) & (X <= self.u)] += self._skewgaussian( X[(X >= self.l) & (X <= self.u)], *self.fit_params)

        return(res)

    @property
    @_check_fit
    def area(self):
        # qres = quad(self, self.l, self.u)
        bl = self._eval_baseline(self.peak_ind)
        qres = trapz(self.peak_counts - bl, self.peak_ind)
        return (qres, 1)

    @property
    @_check_fit
    def centroid(self):
        return self.fit_params[1]

    @property
    @_check_fit
    def amplitude(self):
        return self.fit_params[0]

    @property
    @_check_fit
    def sigma(self):
        return 1. / self.fit_params[2]

    @property
    @_check_fit
    def fwhm(self):
        return 0.939 * self.area[0] / self.amplitude

    def plot(self, baseline=True):
        if baseline:
            plot(self.ind, self.counts, 'b', label="Data", alpha=0.5, linewidth=2)
        else:
            plot(self.ind, self.counts - self._eval_baseline(self.ind), 'b', label="Data", alpha=0.5, linewidth=2)
        plot(self.ind, self(self.ind, with_baseline=baseline), 'r--', label="Fit", alpha=0.5, linewidth=2)
        axvline(self.centroid, color='black', label="Centroid", alpha=0.5)
        legend()
