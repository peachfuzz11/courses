import numpy


class IRLS:
    GAUSS = 'GAUSS'
    LAPLACE = 'LAPLACE'
    HUBER = 'HUBER'
    TUKEYS = 'TUKEYS'
    loss_functions = [
        (GAUSS, 'GAUSS'),
        (LAPLACE, 'LAPLACE'),
        (HUBER, 'HUBER'),
        (TUKEYS, 'TUKEYS'),
    ]

    def __init__(self, data: numpy.ndarray, loss_function: loss_functions, c_breakpoint=None):
        if c_breakpoint is None and (loss_function is IRLS.HUBER or loss_function is IRLS.TUKEYS):
            raise AttributeError

        self.data = data
        self.data_length = len(data)
        self.breakpoint = c_breakpoint

        self.weights = None
        self.residuals = None
        self.mean = None
        self.reset()
        self.data_mean = numpy.mean(data)
        self.data_median = numpy.median(data)
        self.type = loss_function

        self.dispersion_param = 1
        #self.use_standard_deviation_dispersion()  # Default to this

    def use_standard_deviation_dispersion(self):
        self.dispersion_param = numpy.sqrt(1 / self.data_length * numpy.sum(numpy.square(self.data - self.data_mean)))

    def use_mean_absolute_deviation_dispersion(self):
        self.dispersion_param = numpy.sum(numpy.abs(self.data - self.data_median)) / self.data_length

    def iterate(self, n: int):
        for i in range(n):
            self.update()

    def percent_of_median(self, p: float) -> int:
        i = 1
        self.update()
        while True if len(self.mean) < 2 else 1-self.mean[-2] / self.mean[-1] <= p:
            i += 1
            self.update()
            if i > 1000:
                return i
            if self.mean[-1] == self.mean[-2]: break
        return i

    def update(self):
        self.mean.append(self.calculate_mean())
        self.residuals = self.calculate_residual()
        self.update_weights()

    def calculate_mean(self) -> float:
        return numpy.sum(self.weights * self.data) / numpy.sum(self.weights)

    def calculate_residual(self) -> numpy.ndarray:
        return (self.data - self.mean[-1]) / self.dispersion_param

    def update_weights(self):
        if self.type == IRLS.GAUSS:
            pass
        if self.type == IRLS.LAPLACE:
            self.weights = 1.0 / numpy.abs(self.residuals)
        if self.type == IRLS.HUBER:
            top = self.residuals > self.breakpoint
            mid = numpy.abs(self.residuals) <= self.breakpoint
            bot = self.residuals < -1 * self.breakpoint
            self.weights[top] = self.breakpoint / self.residuals[top]
            self.weights[mid] = 1
            self.weights[bot] = -1 * self.breakpoint / self.residuals[bot]
        if self.type == IRLS.TUKEYS:
            top = numpy.abs(self.residuals) <= self.breakpoint
            bot = numpy.abs(self.residuals) > self.breakpoint
            self.weights[top] = numpy.square(1 - numpy.square(self.residuals[top] / self.breakpoint))
            self.weights[bot] = 0

    def get_mean(self) -> []:
        return self.mean

    def get_residuals(self) -> numpy.ndarray:
        return self.residuals

    def get_weights(self) -> numpy.ndarray:
        return self.weights

    def reset(self):
        self.weights = numpy.ones(self.data_length)
        self.residuals = numpy.zeros(self.data_length)
        self.mean = []

    def get_type(self):
        return self.type
