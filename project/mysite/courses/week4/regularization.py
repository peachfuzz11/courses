import numpy


class Regularization:
    LS = 'ls'
    TIKH = 'tikh'
    SM_SPLINE = 'sm'

    def __init__(self, d, year, knots, B, norm=LS, alpha=None):
        self.d = d
        self.B = B
        self.G = self.B.collmat(year)  # G matrix collocation matrix for function value at sites tau
        self.GTG = self.G.T @ self.G
        self.LTL = None
        self.alpha = alpha
        self.knots = knots
        self.norm = norm
        if self.norm == Regularization.SM_SPLINE:
            coll_2 = self.B.collmat(self.knots, deriv_order=2)  # collocation matrix for second derivative at knots
            self.LTL = coll_2.T @ coll_2  # Regularization matrix L^T L for smoothing spline model
        else:
            self.LTL = numpy.identity(len(self.GTG))
        self.model = None

    def system_solve(self):
        if self.alpha is None:
            if self.norm == Regularization.LS:
                self.alpha = 0
            else:
                raise ValueError('Alpha is not given and has not been calculated')
        self.model = self.solve(alpha=self.alpha)
        return self

    def solve(self, alpha):
        return numpy.linalg.lstsq(self.GTG + alpha * self.LTL, self.G.T @ self.d, rcond=None)[0]

    def get_model_params(self):
        return self.model

    def get_misfit(self, alpha):
        m_alpha = self.solve(alpha)
        misfit_norm = numpy.transpose(self.d - self.G @ m_alpha) @ (self.d - self.G @ m_alpha)
        return misfit_norm

    def calculate_alpha_discrepancy_principle(self):
        alphas = [100]
        n_sigma = len(self.d) * numpy.nanvar(self.d)
        misfits = []
        misfits.append(self.get_misfit(alphas[-1]) - n_sigma)
        for i in range(1000):
            alphas.append(alphas[0] - i / 10)
            misfits.append(self.get_misfit(alphas[-1]) - n_sigma)
        return alphas, misfits

    def calculate_alpha_knee(self):
        alphas = [100]
        misfits = []
        misfits.append(self.get_misfit(alphas[-1]))
        for i in range(1000):
            alphas.append(alphas[0] - i / 10)
            misfits.append(self.get_misfit(alphas[-1]))
        return alphas, misfits

    def calculate_gcv(self):

