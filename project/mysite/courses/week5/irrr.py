import numpy

from courses.week1.irls import IRLS


class IRRR:
    LS = 'ls'
    TIKH = 'tikh'

    CONVERGED = False

    def __init__(self, d, G, L1=None, L2=None, norm=1, alpha1=None, alpha2=None, weight_type=IRLS.GAUSS, sigma=12,
                 truncate=True, max_iterations=8):
        self.weight_type = weight_type
        self.d = d  # data, shape n,1
        self.G = G  # design, shape x,y
        n, m = numpy.shape(G)
        self.data_length = n
        self.model_length = m
        self.L1 = L1
        # self.L2 = L2 if L2 is not None else numpy.identity(self.model_length)
        self.W = numpy.identity(self.data_length)
        self.W_m1 = numpy.identity(m) if L1 is None else numpy.identity(L1.shape[0])
        # self.W_m2 = numpy.identity(self.model_length)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.norm = norm
        self.model = None
        self.models = []
        self.iterations = -1
        self.residuals = None
        self.epsilon = 0.0001
        self.sigma = sigma
        self.truncate = truncate
        self.MAX_ITERATIONS = max_iterations

        self.converged_iteration = None
        self.converged_model = None
        self.misfit_norms = []
        self.model_norms = []
        self.rms_misfits = []

    def converge(self) -> 'IRRR':
        conv = 1
        while conv > 10 ** (-2):
            if self.iterations >= self.MAX_ITERATIONS: break
            if self.truncate and self.CONVERGED: break
            if len(self.models) > 1:
                conv = 100 * numpy.linalg.norm(self.models[-2] - self.models[-1]) / numpy.linalg.norm(
                    self.models[-1])
            self.update()
        if self.converged_iteration is None:
            self.converged_iteration = self.iterations
            self.converged_model = self.model
        return self

    def update(self):
        model = self.update_model()
        residuals = (self.d - self.G @ model) / self.sigma

        if not self.truncate:# Not used except for diagnostic in very unstable solutions that overshoot convergence
            rms_misfit = numpy.sqrt((residuals.T @ self.W @ residuals) / self.data_length)
            print(rms_misfit)

            if len(self.rms_misfits) > 0 and rms_misfit >= self.rms_misfits[-1]:
                self.CONVERGED = True
                if self.converged_iteration is None:
                    self.converged_iteration = self.iterations
                    self.converged_model = self.model
                print('Divergent convergence, stopped at iteration: ' + str(self.iterations))
                return self
            elif self.CONVERGED:
                return self

        self.model = model
        self.residuals = residuals
        self.misfit_norms.append(self.get_misfit_norm())
        self.model_norms.append(self.get_model_norm())
        self.rms_misfits.append(self.get_rms_misfit())
        self.models.append(self.model)

        self.iterations += 1
        if self.L1 is None and self.weight_type == IRLS.GAUSS and self.norm == 2:
            self.CONVERGED = True
            return self
        if self.iterations == self.MAX_ITERATIONS: return self  # skip updating weights in last iter so post calls to norms are OK
        self.update_data_weights()
        self.update_model_weights()
        return self

    def update_model(self):
        return self.get_design_alpha() @ self.d

    def get_design_alpha(self):
        if self.alpha1 is None:
            a1 = numpy.zeros((self.model_length, self.model_length))
        elif self.L1 is None:
            a1 = (self.alpha1 * (numpy.identity(self.model_length).T @ self.W_m1 @ numpy.identity(self.model_length)))
        else:
            a1 = (self.alpha1 * (self.L1.T @ self.W_m1 @ self.L1))
        a2 = 0  # (self.alpha2 * (self.L2.T @ self.W_m2 @ self.L2)) if self.alpha2 is not None else numpy.zeros(
        # (self.model_length, self.model_length))
        return numpy.linalg.lstsq(self.G.T @ self.W @ self.G + a1 + a2, self.G.T @ self.W, rcond=None)[
            0]

    def get_resolution(self):
        if self.alpha1 is None:
            a1 = numpy.zeros((self.model_length, self.model_length))
        elif self.L1 is None:
            a1 = (self.alpha1 * (numpy.identity(self.model_length).T @ self.W_m1 @ numpy.identity(self.model_length)))
        else:
            a1 = (self.alpha1 * (self.L1.T @ self.W_m1 @ self.L1))
        a2 = 0  # (self.alpha2 * (self.L2.T @ self.W_m2 @ self.L2)) if self.alpha2 is not None else numpy.zeros(
        # (self.model_length, self.model_length))
        return numpy.linalg.lstsq(self.G.T @ self.W @ self.G + a1 + a2, self.G.T @ self.W @ self.G, rcond=None)[
            0]

    def get_model(self):
        return self.model

    def get_model_norm(self):
        return self.model.T @ self.L1.T @ self.W_m1 @ self.L1 @ self.model if self.L1 is not None else self.model.T @ self.W_m1 @ self.model

    def get_misfit_norm(self):
        misfit_norm = self.residuals.T @ self.W @ self.residuals
        return misfit_norm

    def get_rms_misfit(self):
        misfit_norm = numpy.sqrt(self.get_misfit_norm() / self.data_length)
        return misfit_norm

    @staticmethod
    def get_log_spaced_alphas(n):
        return numpy.geomspace(0.001, 1000, num=n)

    def update_model_weights(self):
        if self.norm == 2: return
        if self.L1 is not None:
            w_m1 = numpy.power(numpy.square(self.L1 @ self.model) + numpy.square(self.epsilon), (self.norm / 2) - 1)
        else:
            w_m1 = numpy.power(numpy.square(self.model) + numpy.square(self.epsilon), (self.norm / 2) - 1)
        # w_m2 = numpy.power(numpy.square(self.L2 @ self.m_alpha) + numpy.square(self.epsilon), (self.norm / 2) - 1)
        self.W_m1 = numpy.diag(w_m1)
        # self.W_m2 = numpy.diag(w_m2)

    def update_data_weights(self):
        if self.weight_type == IRLS.GAUSS:
            return
        if self.weight_type == IRLS.LAPLACE:
            a = 1.0 / numpy.abs(self.residuals)
            self.W = numpy.identity(len(a)) * a
        if self.weight_type == IRLS.HUBER:  # TODO do this in Chris way
            b = 1.365
            top = self.residuals > b
            mid = numpy.abs(self.residuals) <= b
            bot = self.residuals < -1 * b
            a = numpy.ones(self.residuals.shape)
            a[top] = b / self.residuals[top]
            a[mid] = 1
            a[bot] = -1 * b / self.residuals[bot]
            self.W = numpy.identity(len(a)) * a
        if self.weight_type == IRLS.TUKEYS:
            b = 4.685
            a = numpy.ones(self.data_length)
            top = numpy.abs(self.residuals) <= b
            bot = numpy.abs(self.residuals) > b
            a[top] = numpy.square(1 - numpy.square(self.residuals[top] / b))
            a[bot] = 0
            self.W = numpy.identity(len(a)) * a
