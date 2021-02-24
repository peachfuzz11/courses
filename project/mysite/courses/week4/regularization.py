import numpy


class Regularization:
    LS = 'ls'
    TIKH = 'tikh'
    SM_SPLINE = 'sm'

    def __init__(self, d, year, knots, B, norm=LS, alpha=None):
        self.d = d
        self.B = B
        self.year = year
        self.G = self.B.collmat(self.year)  # G matrix collocation matrix for function value at sites tau
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

    def gcv_trace(self):
        GG_alpha = numpy.linalg.lstsq(self.GTG + self.alpha * self.LTL, self.G.T, rcond=None)[0]
        return numpy.trace(numpy.identity(len(self.d)) - self.G @ GG_alpha)

    def get_model_params(self):
        return self.model

    def get_misfit(self, alpha):
        m_alpha = self.solve(alpha)
        misfit_norm = numpy.transpose(self.d - self.G @ m_alpha) @ (self.d - self.G @ m_alpha)
        return misfit_norm

    def calculate_alpha_discrepancy_principle(self):
        alphas = self.guess_alphas()
        n_sigma = len(self.d) * numpy.nanvar(self.d)
        misfits = []
        for alpha in alphas:
            misfits.append(self.get_misfit(alpha) - n_sigma)
        return alphas, misfits

    def calculate_alpha_knee(self):
        alphas = self.guess_alphas()
        misfits = []
        model_norm = []
        for alpha in alphas:
            model_norm.append(self.solve(alpha).T @ self.LTL @ self.solve(alpha))
            misfits.append(self.get_misfit(alpha))
        return alphas, misfits, model_norm

    def calculate_gcv(self):
        alphas = self.guess_alphas()[::20]
        gcv = numpy.zeros(len(alphas))
        for i in range(len(self.d)):
            datum = self.d[i]
            datum_year = self.year[i]
            d = numpy.delete(self.d, i)
            year = numpy.delete(self.year, i)
            G_k = self.B.collmat(datum_year)
            for j in range(len(alphas)):
                reg = Regularization(d, year, self.knots, self.B, self.norm, alphas[j]).system_solve()
                m_alpha = reg.get_model_params()
                misfit = numpy.square((G_k @ m_alpha - datum) / reg.gcv_trace())
                # trace = self.gcv_trace(alphas[j])
                gcv[j] += misfit
        return gcv, alphas

    def guess_alphas(self):
        return numpy.geomspace(0.001, 1000, num=1000)
