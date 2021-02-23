import numpy
from matplotlib import pyplot

from courses.week1.irls import IRLS
from courses.week2.robust import Robust
from main.helpers.figurehelper import Figurehelper


class DataHelper:

    @staticmethod
    def create_g_matrix(time: numpy.ndarray, n=6):
        G_daily = numpy.ones(len(time))
        for n in numpy.arange(1, 6 + 1):
            G_daily = numpy.vstack((G_daily, numpy.cos(n * 2 * numpy.pi * time), numpy.sin(n * 2 * numpy.pi * time)))
        return G_daily.T

    @staticmethod
    def get_function(g, m):
        n, s = numpy.shape(g)
        f = numpy.zeros(n)
        for i in range(n):
            f[i] = numpy.sum(g[i, :] * m)
        return f

    @staticmethod
    def remove_nans_from_data(boulder_data, boulder_nans, kp_data, t, tseries):
        boulder_data_nanless = boulder_data[boulder_nans]
        boulder_time_nanless = t[boulder_nans]
        boulder_time_series_nanless = tseries[boulder_nans]
        boulder_kp_data = kp_data[boulder_nans]
        boulder_g = DataHelper.create_g_matrix(boulder_time_nanless)
        return boulder_data_nanless, boulder_g, boulder_time_series_nanless, boulder_kp_data

    @staticmethod
    def apply_irls(boulder_data_nanless, boulder_g):
        boulder_laplace = Robust(boulder_data_nanless, boulder_g, IRLS.LAPLACE).converge()
        boulder_huber = Robust(boulder_data_nanless, boulder_g, IRLS.HUBER, 1.345).converge()
        boulder_tukey = Robust(boulder_data_nanless, boulder_g, IRLS.TUKEYS, 4.685).converge()
        return boulder_huber, boulder_laplace, boulder_ls, boulder_tukey

    @staticmethod
    def plot_functions(boulder_data_nanless, boulder_time_series_nanless, f_huber, f_lap, f_ls, f_tuker):
        fig1, ax1 = pyplot.subplots(figsize=(12, 6))
        ax1.plot(boulder_time_series_nanless, f_ls, color='blue', alpha=0.3, marker='.', label='ls')
        ax1.plot(boulder_time_series_nanless, f_lap, color='yellow', alpha=0.3, marker='.', label='laplace')
        ax1.plot(boulder_time_series_nanless, f_huber, color='red', alpha=0.3, marker='.', label='huber')
        ax1.plot(boulder_time_series_nanless, f_tuker, color='green', alpha=0.3, marker='.', label='tukeys')
        ax1.plot(boulder_time_series_nanless, boulder_data_nanless, color='black', alpha=0.5, label='data')
        ax1.legend(loc='upper left')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Magnetic component')
        img1 = Figurehelper(fig1).to_png()
        return img1

    @staticmethod
    def estimate_function_from_model(g, huber, laplace, ls, tukey):
        f_ls = DataHelper.get_function(g, ls.get_model_vector())
        f_lap = DataHelper.get_function(g, laplace.get_model_vector())
        f_huber = DataHelper.get_function(g, huber.get_model_vector())
        f_tuker = DataHelper.get_function(g, tukey.get_model_vector())
        return f_huber, f_lap, f_ls, f_tuker
