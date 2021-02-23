import numpy

from courses.week1.irls import IRLS


class Robust:

    def __init__(self, data_vector: numpy.ndarray, g: numpy.ndarray, lf: IRLS.loss_functions, c_breakpoint=1.5):
        if c_breakpoint is None and (type is IRLS.HUBER or type is IRLS.TUKEYS):
            raise AttributeError

        self.data_vector = data_vector
        self.model_vector = None
        self.g = g
        self.data_length = len(self.data_vector)

        self.weights = numpy.eye(self.data_length, self.data_length)
        self.type = lf
        self.residuals = numpy.zeros(self.data_length)
        self.breakpoint = c_breakpoint

        self.models = []
        self.iterations = -1

    def update(self):
        self.update_model()
        self.update_residuals()
        self.update_weights()
        self.models.append(self.model_vector)
        self.iterations += 1
        return self

    def converge(self, p=1) -> 'Robust':
        while True:
            if self.iterations > 15: break
            if len(self.models) > 1:
                diff = numpy.abs(self.models[-2] / self.models[-1] - 1)
                if numpy.sum(diff) / len(diff) <= p / 100:
                    break

            try:
                self.update()
            except numpy.linalg.LinAlgError as e:
                print(self.type, self.iterations)
                print(numpy.dot(numpy.dot(numpy.transpose(self.g), self.weights), self.g))
                print('traceback', e)
                return self

        print(self.type, self.iterations)
        print(self.model_vector)
        return self

    def update_model(self):
        self.model_vector = \
        numpy.linalg.lstsq((self.g.T @ self.weights @ self.g), (self.g.T @ self.weights @ self.data_vector),
                           rcond=None)[0]

    def update_residuals(self):
        self.residuals = self.data_vector - numpy.dot(self.g, self.model_vector)

    def update_weights(self):
        if self.type == IRLS.GAUSS:
            pass
        if self.type == IRLS.LAPLACE:
            a = 1.0 / numpy.abs(self.residuals)
            self.weights = numpy.eye(self.data_length, self.data_length) * a
        if self.type == IRLS.HUBER:
            top = self.residuals > self.breakpoint
            mid = numpy.abs(self.residuals) <= self.breakpoint
            bot = self.residuals < -1 * self.breakpoint
            a = numpy.ones(self.data_length)
            a[top] = self.breakpoint / self.residuals[top]
            a[mid] = 1
            a[bot] = -1 * self.breakpoint / self.residuals[bot]
            self.weights = numpy.eye(self.data_length, self.data_length) * a
        if self.type == IRLS.TUKEYS:
            a = numpy.ones(self.data_length)
            top = numpy.abs(self.residuals) <= self.breakpoint
            bot = numpy.abs(self.residuals) > self.breakpoint
            a[top] = numpy.square(1 - numpy.square(self.residuals[top] / self.breakpoint))
            a[bot] = 0
            self.weights = numpy.eye(self.data_length, self.data_length) * a

    def get_model_vectors(self) -> []:
        return self.models

    def get_model_vector(self) -> numpy.ndarray:
        return self.model_vector
