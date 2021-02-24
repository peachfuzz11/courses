import numpy
from bspline import splinelab, bspline
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import ensure_csrf_cookie
from matplotlib import pyplot

from courses.data.dataloader import CourseDataLoader
from courses.week1.irls import IRLS
from courses.week1.modelHelper import ModelHelper
from courses.week2.robust import Robust
from courses.week3.data_helper import DataHelper
from courses.week4.regularization import Regularization
from main.helpers.figurehelper import Figurehelper


@ensure_csrf_cookie
def courses(request):
    return render(request, 'main/courses.html', {
    })


def exercise_3_1(request):
    month, year, anomaly, standardized = CourseDataLoader.load_soi_txt()

    anomaly_yr_mean = numpy.mean(anomaly, axis=1)
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(year, anomaly_yr_mean, 'or', label="Annual means", fillstyle='none', markersize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("SOI index")
    ax.set_title("Annual means of Southern Oscillation Index")
    ax.set_xticks(numpy.arange(1950, 2015 + 5, 5))
    ax.grid()
    ax.legend()
    img1 = Figurehelper(fig).to_png()

    # 10 year running average
    from scipy.ndimage.filters import uniform_filter1d
    # A 1D uniform filter replaces the value of a point by the mean value of the range of values centered at the point
    # Size determines the range, e.g. size=11 is a mean over the central point and 5 points on each side
    anomaly_10yr_mean = uniform_filter1d(anomaly_yr_mean, size=11, mode="nearest")
    "# Remove data for 1970-1975\n",
    i_rm = numpy.logical_or(year > 1975, year < 1970)
    year_rm = year[i_rm]
    anomaly_10yr_mean_rm = anomaly_10yr_mean[i_rm]

    # Plot
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(year, anomaly_10yr_mean, 'ok', label="Removed", fillstyle='none', markersize=10)
    ax.plot(year_rm, anomaly_10yr_mean_rm, 'or', label="Annual running means", fillstyle='none', markersize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("SOI index")
    ax.set_title("10yr running mean of Southern Oscillation Index")
    ax.set_xticks(numpy.arange(1950, 2015 + 5, 5))
    ax.grid()
    ax.legend()
    img2 = Figurehelper(fig).to_png()

    # order of spline, cubic = 4\n
    step = 2
    knots = numpy.arange(1951 - step, 2015 + 2 * step, step)
    # Predictions
    spline_order = 4
    k = splinelab.augknt(knots, spline_order)  # add endpoint repeats as appropriate for spline order p
    B = bspline.Bspline(k, spline_order)  # create spline basis of order p on knots k

    m_lsq = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B,
                           norm=Regularization.LS).system_solve()
    m_tikh = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.TIKH,
                            alpha=1).system_solve()
    m_sm_spline = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.SM_SPLINE,
                                 alpha=0.1).system_solve()
    xx = numpy.linspace(1951, 2015, 1000)

    y0 = numpy.array([numpy.sum(B(x) * m_lsq.get_model_params()) for x in xx])
    y1 = numpy.array([numpy.sum(B(x) * m_tikh.get_model_params()) for x in xx])
    y2 = numpy.array([numpy.sum(B(x) * m_sm_spline.get_model_params()) for x in xx])
    fig3, ax3 = pyplot.subplots(figsize=(12, 6))
    ax3.plot(year, anomaly_10yr_mean, 'ok', label="Removed", fillstyle='none', markersize=10)
    ax3.plot(year_rm, anomaly_10yr_mean_rm, 'or', label="Annual running means", fillstyle='none', markersize=10)
    ax3.plot(xx, y0, '-', label='Lsq spline')
    ax3.plot(xx, y1, '-', label='Tikh spline alpha=1')
    ax3.plot(xx, y2, '-', label='SM spline alpha=0.1')
    ax3.set_xlabel("Year")
    ax3.set_ylabel("SOI index")
    ax3.set_title("Southern Oscillation Index models (data for 1970-1975 removed)")
    ax3.set_xticks(numpy.arange(1950, 2015 + 5, 5))
    ax3.grid()

    tikh_alphas, tikh_misfit = m_tikh.calculate_alpha_discrepancy_principle()
    tikh_alpha = tikh_alphas[numpy.argmin(numpy.abs(tikh_misfit))]
    sm_alphas, sm_misfit = m_sm_spline.calculate_alpha_discrepancy_principle()
    sm_alpha = sm_alphas[numpy.argmin(numpy.abs(sm_misfit))]

    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(tikh_alphas, numpy.square(tikh_misfit), '-', label='Tikh alpha= ' + str(numpy.round(tikh_alpha, 1)))
    ax.plot(sm_alphas, numpy.square(sm_misfit), '-', label='SM alpha= ' + str(numpy.round(sm_alpha, 1)))
    # ax.plot(xx, y1, '-', label='Tikh spline')
    # ax.plot(xx, y2, '-', label='SM spline')
    ax.set_ylabel("Misfit norm")
    ax.set_xlabel("Alpha")
    ax.set_title("Discrepancy principle")
    ax.legend(loc="upper left")
    img4 = Figurehelper(fig).to_png()

    tikh_alphas, tikh_misfit, tikh_model_norm = m_tikh.calculate_alpha_knee()
    tikh_curvature = numpy.abs(numpy.gradient(numpy.gradient(tikh_misfit))) + numpy.abs(
        numpy.gradient(numpy.gradient(tikh_model_norm)))
    tikh_alpha = tikh_alphas[numpy.argmax(tikh_curvature)]
    tikh_alpha = numpy.round(numpy.sqrt(tikh_alpha), 2)
    sm_alphas, sm_misfit, sm_model_norm = m_sm_spline.calculate_alpha_knee()
    sm_curvature = numpy.abs(numpy.gradient(numpy.gradient(sm_misfit))) + numpy.abs(
        numpy.gradient(numpy.gradient(sm_model_norm)))
    sm_alpha = sm_alphas[numpy.argmax(sm_curvature)]
    sm_alpha = numpy.round(numpy.sqrt(sm_alpha), 2)

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(tikh_misfit, tikh_model_norm, '-', label='Tikh alpha= ' + str(tikh_alpha))
    ax.plot(sm_misfit, sm_model_norm, '-', label='SM alpha= ' + str(sm_alpha))
    # ax.plot(xx, y1, '-', label='Tikh spline')
    # ax.plot(xx, y2, '-', label='SM spline')
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm m_a.T@LTL@m_a")
    ax.set_title("Knee - model norm vs misfit")
    ax.legend(loc="upper left")
    img5 = Figurehelper(fig).to_png()

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(tikh_alphas, tikh_curvature, '-', label='Tikh alpha= ' + str(tikh_alpha))
    ax.plot(sm_alphas, sm_curvature, '-', label='SM alpha= ' + str(sm_alpha))
    ax.set_xscale('log')
    ax.set_xlabel("alpha")
    ax.set_ylabel("Curvature")
    ax.set_title("Curvature vs alpha")
    ax.legend(loc="upper left")
    img6 = Figurehelper(fig).to_png()

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(tikh_alphas, tikh_model_norm, '-', label='Tikh alpha= ' + str(tikh_alpha))
    ax.plot(sm_alphas, sm_model_norm, '-', label='SM alpha= ' + str(sm_alpha))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Model norm")
    ax.set_title("Model norm vs alpha")
    ax.legend(loc="upper left")
    img7 = Figurehelper(fig).to_png()

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(tikh_alphas, tikh_misfit, '-', label='Tikh alpha= ' + str(tikh_alpha))
    ax.plot(sm_alphas, sm_misfit, '-', label='SM alpha= ' + str(sm_alpha))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Misfit norm")
    ax.set_title("misfit vs alpha")
    ax.legend(loc="upper left")
    img8 = Figurehelper(fig).to_png()

    fig, ax = pyplot.subplots(figsize=(12, 6))
    tikh_gcv, tikh_alphas = m_tikh.calculate_gcv()
    tikh_gcv_alpha = numpy.round(tikh_alphas[numpy.argmin(tikh_gcv)])
    sm_gcv, sm_alphas = m_sm_spline.calculate_gcv()
    sm_gcv_alpha = sm_alphas[numpy.argmin(tikh_gcv)]

    ax.plot(tikh_alphas, tikh_gcv, '-', label='Tikh alpha= ' + str(numpy.round(tikh_gcv_alpha, 2)))
    ax.plot(sm_alphas, sm_gcv, '-', label='SM alpha= ' + str(numpy.round(sm_gcv_alpha, 2)))
    ax.set_xlabel("alpha")
    ax.set_ylabel("gcv")
    ax.set_xscale('log')
    ax.set_title("GCV")
    ax.legend(loc="upper left")
    img9 = Figurehelper(fig).to_png()

    m_tikh_desc = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.TIKH,
                                 alpha=tikh_alpha).system_solve()
    m_sm_spline_desc = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.SM_SPLINE,
                                      alpha=sm_alpha).system_solve()
    m_tikh_gcv = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.TIKH,
                                alpha=tikh_gcv_alpha).system_solve()
    m_sm_spline_gcv = Regularization(anomaly_10yr_mean_rm, year_rm, knots, B, norm=Regularization.SM_SPLINE,
                                     alpha=sm_gcv_alpha).system_solve()
    y4 = numpy.array([numpy.sum(B(x) * m_tikh_desc.get_model_params()) for x in xx])
    y5 = numpy.array([numpy.sum(B(x) * m_sm_spline_desc.get_model_params()) for x in xx])
    y6 = numpy.array([numpy.sum(B(x) * m_tikh_gcv.get_model_params()) for x in xx])
    y7 = numpy.array([numpy.sum(B(x) * m_sm_spline_gcv.get_model_params()) for x in xx])
    ax3.plot(xx, y4, '-', label='tikh desc alpha=' + str(numpy.round(tikh_alpha, 2)))
    ax3.plot(xx, y5, '-', label='SM desc alpha=' + str(numpy.round(sm_alpha, 2)))
    ax3.plot(xx, y6, '-', label='tikh gcv alpha=' + str(numpy.round(tikh_gcv_alpha, 2)))
    ax3.plot(xx, y7, '-', label='SM gcv alpha=' + str(numpy.round(sm_gcv_alpha, 2)))
    ax3.legend(loc="upper left")
    img3 = Figurehelper(fig3).to_png()

    return render(request, 'main/exercise_3_1.html', {
        'img1': img1,
        'img2': img2,
        'img3': img3,
        'img4': img4,
        'img5': img5,
        'img6': img6,
        'img7': img7,
        'img8': img8,
        'img9': img9,
    })


def exercise_2_3(request):
    data = CourseDataLoader.load_mat('B_X_2008.mat')

    boulder_data = data['X_BOU']
    boulder_data = boulder_data - numpy.nanmean(boulder_data)
    alaska_data = data['X_CMO']
    alaska_data = alaska_data - numpy.nanmean(alaska_data)
    kp_data = data['Kp']
    t_daily = numpy.arange(0, 1 + 1 / 48, 1 / 48)  # one day
    t_comp = numpy.arange(1, 24 + 1)
    t_comp = t_comp - 0.5
    t = data['t']
    ts = t * 24 * 60 * 60
    epoch = numpy.array('2000-01-01', dtype='datetime64[s]')  # Define starting epoch
    tseries = epoch + ts.astype(int)  # Generate time-series for plotting"
    G_daily = DataHelper.create_g_matrix(t_daily)

    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(tseries, alaska_data, color='blue', label='Alaska')
    ax.plot(tseries, boulder_data, color='red', label='Boulder')
    ax.plot(tseries, kp_data, color='green', label='global mean')
    ax.set_title('Alaska and Boulder data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Magnetic component')
    ax.legend(loc='upper left')
    img = Figurehelper(fig).to_png()

    boulder_nans = numpy.isnan(boulder_data) != 1
    boulder_data_nanless = boulder_data[boulder_nans]
    boulder_time_nanless = t[boulder_nans]
    boulder_time_series_nanless = tseries[boulder_nans]
    boulder_kp_data = kp_data[boulder_nans]
    boulder_g = DataHelper.create_g_matrix(boulder_time_nanless)

    boulder_mean_data = numpy.nanmean(boulder_data.reshape(-1, 24), axis=0)
    boulder_sub = boulder_data.copy()
    boulder_sub[kp_data > 20] = numpy.nan
    boulder_mean_data_sub = numpy.nanmean(boulder_sub.reshape(-1, 24), axis=0)

    boulder_ls = Robust(boulder_data_nanless, boulder_g, IRLS.GAUSS).converge().get_model_vector()
    boulder_laplace = Robust(boulder_data_nanless, boulder_g, IRLS.LAPLACE).converge().get_model_vector()
    boulder_huber = Robust(boulder_data_nanless, boulder_g, IRLS.HUBER, 1.345).converge().get_model_vector()
    # boulder_tukey = Robust(boulder_data_nanless, boulder_g, IRLS.TUKEYS, 4.685).converge().get_model_vector()
    f_ls = G_daily @ boulder_ls
    f_lap = G_daily @ boulder_laplace
    f_huber = G_daily @ boulder_huber
    # f_tukey = DataHelper.get_function(G_daily, boulder_tukey.get_model_vector())

    fig1, ax1 = pyplot.subplots(figsize=(12, 6))
    ax1.plot(t_daily * 24, f_ls, color='blue', alpha=0.3, marker='.', label='ls')
    ax1.plot(t_daily * 24, f_lap, color='yellow', alpha=0.3, marker='.', label='laplace')
    ax1.plot(t_daily * 24, f_huber, color='red', alpha=0.3, marker='.', label='huber')
    # ax1.plot(t_daily, f_tukey, color='green', alpha=0.3, marker='.', label='tukeys')
    ax1.plot(t_comp, boulder_mean_data, color='black', alpha=0.5, label='data')
    ax1.plot(t_comp, boulder_mean_data_sub, color='magenta', alpha=0.5, label='data_sub')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Magnetic component')
    img1 = Figurehelper(fig1).to_png()

    if False:
        boulder_nans = numpy.logical_and(numpy.isnan(boulder_data) != 1, numpy.abs(kp_data) < 20)
        boulder_data_nanless, boulder_g, boulder_time_series_nanless, boulder_kp = DataHelper.remove_nans_from_data(
            boulder_data,
            boulder_nans,
            kp_data, t, tseries)
        boulder_huber, boulder_laplace, boulder_ls, boulder_tukey = DataHelper.apply_irls(boulder_data_nanless,
                                                                                          boulder_g)
        f_huber, f_lap, f_ls, f_tuker = DataHelper.estimate_function_from_model(boulder_g, boulder_huber,
                                                                                boulder_laplace,
                                                                                boulder_ls,
                                                                                boulder_tukey)
        img11 = DataHelper.plot_functions(boulder_data_nanless, boulder_time_series_nanless, f_huber, f_lap, f_ls,
                                          f_tuker)

        alaska_nans = numpy.logical_and(numpy.isnan(alaska_data) != 1, numpy.abs(alaska_data) <= 20)
        alaska_data_nanless, alaska_g, alaska_time_series_nanless, alaska_kp = DataHelper.remove_nans_from_data(
            alaska_data,
            alaska_nans,
            kp_data, t,
            tseries)

        alaska_huber, alaska_laplace, alaska_ls, alaska_tukey = DataHelper.apply_irls(alaska_data_nanless, alaska_g)
        f_huber, f_lap, f_ls, f_tuker = DataHelper.estimate_function_from_model(alaska_g, alaska_huber, alaska_laplace,
                                                                                alaska_ls,
                                                                                alaska_tukey)
        img2 = DataHelper.plot_functions(alaska_data_nanless, alaska_time_series_nanless, f_huber, f_lap, f_ls, f_tuker)

        alaska_nans = numpy.logical_and(numpy.isnan(alaska_data) != 1, numpy.abs(kp_data) <= 20)
        alaska_data_nanless, alaska_g, alaska_time_series_nanless, alaska_kp = DataHelper.remove_nans_from_data(
            alaska_data,
            alaska_nans,
            kp_data, t,
            tseries)

        alaska_huber, alaska_laplace, alaska_ls, alaska_tukey = DataHelper.apply_irls(alaska_data_nanless, alaska_g)
        f_huber, f_lap, f_ls, f_tuker = DataHelper.estimate_function_from_model(alaska_g, alaska_huber, alaska_laplace,
                                                                                alaska_ls,
                                                                                alaska_tukey)
        img22 = DataHelper.plot_functions(alaska_data_nanless, alaska_time_series_nanless, f_huber, f_lap, f_ls,
                                          f_tuker)

        combined_nans = numpy.logical_or(alaska_nans, boulder_nans)
        combined_data = numpy.hstack((alaska_data[combined_nans], boulder_data[combined_nans]))
        combined_time = numpy.hstack((t[combined_nans], t[combined_nans]))
        combined_kp = numpy.hstack((kp_data[combined_nans], kp_data[combined_nans]))
        combined_g = DataHelper.create_g_matrix(combined_time)

    return render(request, 'main/exercise_2_3.html', {
        'data': img,
        'boulder1': img1,
        # 'boulder2': img11,
        # 'alaska1': img2,
        # 'alaska2': img22,
    })


def week1(request):
    data = numpy.asarray([1, 2, 3, 4, 5, 6, 7, 80, 9])

    viewModels = []
    models = [
        IRLS(data, IRLS.GAUSS),
        IRLS(data, IRLS.LAPLACE),
        IRLS(data, IRLS.HUBER, 1.365),
        IRLS(data, IRLS.TUKEYS, 4.685),
    ]
    for model in models:
        viewModels.append(ModelHelper.getAttribute(model))

    file = CourseDataLoader.load_netcdf('gistemp250_GHCNv4.nc')
    dat_fil = CourseDataLoader.load_dat('T_data.dat')
    # print(file.variables)
    # print(dat_fil)

    return render(request, 'main/week1.html', {
        'models': viewModels,
    })


def week2(request):
    t = numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g = numpy.ones((len(t), 3))
    g[:, 1] = t
    g[:, 2] = numpy.square(t) / 2
    d = numpy.asarray([109.4, 187.5, 267.5, 331.9, 386.1, 428.4, 452.2, 491.1, 512.3, 513])
    d_a = numpy.asarray([109.4, 187.5, 267.5, 631.9, 386.1, 428.4, 452.2, 491.1, 512.3, 513])
    models = []

    r = Robust(d, g, IRLS.GAUSS)
    r.converge()
    figs = ModelHelper.get_robust_plot(r)
    t = {
        'info': 'GAUSS no anomaly',
        'figs': figs,
    }
    models.append(t)

    r2 = Robust(d_a, g, IRLS.GAUSS)
    r2.converge()
    figs2 = ModelHelper.get_robust_plot(r2)
    t = {
        'info': 'GAUSS with anomaly',
        'figs': figs2,
    }
    models.append(t)

    r3 = Robust(d, g, IRLS.HUBER)
    r3.converge()
    figs = ModelHelper.get_robust_plot(r3)
    t = {
        'info': 'HUBER no anomaly',
        'figs': figs,
    }
    models.append(t)

    r4 = Robust(d_a, g, IRLS.HUBER)
    r4.converge()
    figs2 = ModelHelper.get_robust_plot(r4)
    t = {
        'info': 'HUBER with anomaly',
        'figs': figs2,
    }
    models.append(t)

    t = {
        'info COMBINED model predictions',
        ''
    }

    return render(request, 'main/week2.html', {
        'models': models,
    })
