import time

import numpy
from bspline import splinelab, bspline
from chaosmagpy import model_utils
from chaosmagpy.model_utils import design_gauss
from django.shortcuts import render
# Create your views here.
from django.views.decorators.csrf import ensure_csrf_cookie
from matplotlib import pyplot, colors, cm
from numpy.core import cumsum
from scipy.stats import cumfreq

from courses.data.dataloader import CourseDataLoader
from courses.week1.irls import IRLS
from courses.week1.modelHelper import ModelHelper
from courses.week2.robust import Robust
from courses.week3.data_helper import DataHelper
from courses.week4.regularization import Regularization
from courses.week5.irrr import IRRR
from courses.week6.assignment1_helper import plot_global, plot_resolution
from main.helpers.figurehelper import Figurehelper


@ensure_csrf_cookie
def courses(request):
    return render(request, 'main/courses.html', {
    })


def assignment_1_full(request):
    print("Full")
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb_hr = phi_cmb.reshape(-1, )
    theta_cmb_hr = theta_cmb.reshape(-1, )
    r_cmb_hr = c * numpy.ones(theta_cmb_hr.shape)
    Gr_cmb_hr, _, _ = design_gauss(r_cmb_hr, theta_cmb_hr, phi_cmb_hr, N)

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    imgs = []

    irrr_ls = IRRR(d, Gr).converge()
    m_ls = irrr_ls.get_model()
    Br_ls = Gr_cmb_hr @ m_ls
    r_ls = irrr_ls.get_resolution()
    print(irrr_ls.residuals.shape)
    print('LS')
    if True:
        irrr_huber = IRRR(d, Gr, weight_type=IRLS.HUBER).converge()
        m_ls_huber = irrr_huber.get_model()
        Br_ls_huber = Gr_cmb_hr @ m_ls_huber
        r_ls_huber = irrr_huber.get_resolution()
        print('LS_huber')

        irrr_tikh = IRRR(d, Gr, alpha1=0.0336, norm=2).converge()
        m_tikh = irrr_tikh.get_model()
        Br_tikh = Gr_cmb_hr @ m_tikh
        r_tikh = irrr_tikh.get_resolution()
        print('Tikhonov')

        irrr_l2 = IRRR(d, Gr, alpha1=10 ** (-7), L1=L, norm=2).converge()
        m_l2 = irrr_l2.get_model()
        Br_l2 = Gr_cmb_hr @ m_l2
        r_l2 = irrr_l2.get_resolution()
        print('l2')

        irrr_l1 = IRRR(d, Gr, alpha1=10 ** (-7), L1=L, norm=1).converge()
        m_l1 = irrr_l1.get_model()
        Br_l1 = Gr_cmb_hr @ m_l1
        r_l1 = irrr_l1.get_resolution()
        print('l1')

        alpha_l1_disc = 0.00785
        irrr_l1_robust_disc = IRRR(d, Gr, alpha1=alpha_l1_disc, L1=L, norm=1).converge()
        m_l1_disc = irrr_l1_robust_disc.get_model()
        Br_l1_disc = Gr_cmb_hr @ m_l1_disc
        r_l1_disc = irrr_l1_robust_disc.get_resolution()
        print('l1 disc')

        alpha_l1_gcv = 0.048
        irrr_gcv = IRRR(d, Gr, alpha1=alpha_l1_gcv, L1=L, norm=1).converge()
        m_l1_gcv = irrr_gcv.get_model()
        Br_l1_gcv = Gr_cmb_hr @ m_l1_gcv
        r_l1_gcv = irrr_gcv.get_resolution()
        print('l1 disc')

        alpha_l1_knee1 = 3 * 10 ** (-5)
        irrr_l1_robust_knee1 = IRRR(d, Gr, alpha1=alpha_l1_knee1, L1=L, norm=1).converge()
        m_l1_knee1 = irrr_l1_robust_knee1.get_model()
        Br_l1_knee1 = Gr_cmb_hr @ m_l1_knee1
        r_l1_knee1 = irrr_l1_robust_knee1.get_resolution()
        print('l1 knee1')

        alpha_l1_knee2 = 5 * 10 ** (-5)
        irrr_l1_robust_knee2 = IRRR(d, Gr, alpha1=alpha_l1_knee2, L1=L, norm=1).converge()
        m_l1_knee2 = irrr_l1_robust_knee2.get_model()
        Br_l1_knee2 = Gr_cmb_hr @ m_l1_knee2
        r_l1_knee2 = irrr_l1_robust_knee2.get_resolution()
        print('l1 knee2')

        alpha_l1_knee3 = 8 * 10 ** (-5)
        irrr_l1_robust_knee3 = IRRR(d, Gr, alpha1=alpha_l1_knee3, L1=L, norm=1).converge()
        m_l1_knee3 = irrr_l1_robust_knee3.get_model()
        Br_l1_knee3 = Gr_cmb_hr @ m_l1_knee3
        r_l1_knee3 = irrr_l1_robust_knee3.get_resolution()
        print('l1 knee3')

        # TODO GCV

        ps_ls = model_utils.power_spectrum(m_ls, radius=ps_r)
        ps_ls_huber = model_utils.power_spectrum(m_ls_huber, radius=ps_r)
        ps_tikh = model_utils.power_spectrum(m_tikh, radius=ps_r)
        ps_l1 = model_utils.power_spectrum(m_l1, radius=ps_r)
        ps_l2 = model_utils.power_spectrum(m_l2, radius=ps_r)
        ps_l1_irrr_disc = model_utils.power_spectrum(m_l1_disc, radius=ps_r)
        ps_l1_irrr_gcv = model_utils.power_spectrum(m_l1_gcv, radius=ps_r)
        ps_l1_irrr_knee1 = model_utils.power_spectrum(m_l1_knee1, radius=ps_r)
        ps_l1_irrr_knee2 = model_utils.power_spectrum(m_l1_knee2, radius=ps_r)
        ps_l1_irrr_knee3 = model_utils.power_spectrum(m_l1_knee3, radius=ps_r)

        n = numpy.arange(1, N + 1)
        fig, ax = pyplot.subplots(figsize=(12, 6))
        ax.semilogy(n, ps_ls, label="LS")
        ax.semilogy(n, ps_ls_huber, label="LS huber weights")
        ax.semilogy(n, ps_tikh, label="Tikhonov")
        ax.semilogy(n, ps_l1, label="L1")
        ax.semilogy(n, ps_l2, label="L2")
        ax.semilogy(n, ps_l1_irrr_disc, label="R L1 disc")
        ax.semilogy(n, ps_l1_irrr_gcv, label="R L1 gcv")
        ax.semilogy(n, ps_l1_irrr_knee1, label="R L1 knee1")
        ax.semilogy(n, ps_l1_irrr_knee2, label="R L1 knee2")
        ax.semilogy(n, ps_l1_irrr_knee3, label="R L1 knee3")
        ax.set_xlabel("degree n")
        ax.set_ylabel(r"Power [nT$^2$]")
        ax.set_title(r"Power spectra of estimated geomagnetic field at $r={}$km".format(ps_r))
        nmax = numpy.max([len(n)])
        ax.set_xticks(numpy.arange(1, nmax + 1))
        ax.legend()
        ax.grid()
        imgs.append(Figurehelper(fig).to_png())

        ls = plot_global(phi_cmb_hr, theta_cmb_hr, Br_ls, title="Least squares, CMB, nmax={}".format(N))
        ls_huber = plot_global(phi_cmb_hr, theta_cmb_hr, Br_ls_huber,
                               title="Least squares huber weights, CMB, nmax={}".format(N))
        tikh = plot_global(phi_cmb_hr, theta_cmb_hr, Br_tikh, title="Tikhonov, CMB, nmax={}".format(N))
        l1 = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1, title="L1 norm, CMB, nmax={}".format(N))
        l2 = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l2, title="L2 norm, CMB, nmax={}".format(N))
        l1_disc = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1_gcv,
                              title="Robust L1 norm discrepancy, CMB, nmax={}".format(N))
        l1_gcv = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1_disc,
                              title="Robust L1 norm gcv, CMB, nmax={}".format(N))
        l1_knee1 = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1_knee1,
                               title="Robust L1 norm knee1, CMB, nmax={}".format(N))
        l1_knee2 = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1_knee2,
                               title="Robust L1 norm knee2, CMB, nmax={}".format(N))
        l1_knee3 = plot_global(phi_cmb_hr, theta_cmb_hr, Br_l1_knee3,
                               title="Robust L1 norm knee3, CMB, nmax={}".format(N))

        imgs.extend([ls, ls_huber, tikh, l1, l2, l1_disc,l1_gcv, l1_knee1, l1_knee2, l1_knee3])

        res_ls = plot_resolution(r_ls, 'Least squares')
        res_ls_huber = plot_resolution(r_ls_huber, 'Least squares huber weights')
        res_tikh = plot_resolution(r_tikh, 'Tikhonov')
        res_l1 = plot_resolution(r_l1, 'L1')
        res_l2 = plot_resolution(r_l2, 'L2')
        res_l1_disc = plot_resolution(r_l1_disc, 'L1 disc')
        res_l1_gcv = plot_resolution(r_l1_disc, 'L1 gcv')
        res_l1_knee1 = plot_resolution(r_l1_knee1, 'L1 knee1')
        res_l1_knee2 = plot_resolution(r_l1_knee2, 'L1 knee2')
        res_l1_knee3 = plot_resolution(r_l1_knee3, 'L1 knee3')
        imgs.extend(
            [res_ls, res_ls_huber, res_tikh, res_l1, res_l2, res_l1_disc, res_l1_gcv, res_l1_knee1, res_l1_knee2, res_l1_knee3])

    n_bins = int(2 * len(d) ** (-1 / 3) * (
            numpy.median(d[int(len(numpy.sort(irrr_ls.residuals)) / 2) + 1:None]) - numpy.median(
        d[0:int(len(numpy.sort(irrr_ls.residuals)) / 2)])))
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.hist(irrr_ls.residuals, bins=n_bins, label="LS")
    ax.hist(irrr_huber.residuals, bins=n_bins, label="LS huber weights")
    ax.hist(irrr_tikh.residuals, bins=n_bins, label="Tikhonov")
    ax.hist(irrr_l1.residuals, bins=n_bins, label="L1")
    ax.hist(irrr_l2.residuals, bins=n_bins, label="L2")
    ax.hist(irrr_l1_robust_disc.residuals, bins=n_bins, label="R L1 disc")
    ax.hist(irrr_gcv.residuals, bins=n_bins, label="R L1 gcv")
    ax.hist(irrr_l1_robust_knee1.residuals, bins=n_bins, label="R L1 knee1")
    ax.hist(irrr_l1_robust_knee2.residuals, bins=n_bins, label="R L1 knee2")
    ax.hist(irrr_l1_robust_knee3.residuals, bins=n_bins, label="R L1 knee3")
    ax.set_xlabel("n")
    ax.set_ylabel('Residual value')
    ax.set_title('Residual distribution bins=' + str(n_bins))
    ax.legend()
    imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.scatter(1,irrr_ls.get_rms_misfit(), label="LS")
    ax.scatter(2,irrr_huber.get_rms_misfit(), label="LS huber weights")
    ax.scatter(3,irrr_tikh.get_rms_misfit(), label="Tikhonov")
    ax.scatter(4,irrr_l1.get_rms_misfit(), label="L1")
    ax.scatter(5,irrr_l2.get_rms_misfit(), label="L2")
    ax.scatter(6,irrr_l1_robust_disc.get_rms_misfit(), label="R L1 disc")
    ax.scatter(7,irrr_gcv.get_rms_misfit(), label="R L1 gcv")
    ax.scatter(8,irrr_l1_robust_knee1.get_rms_misfit(), label="R L1 knee1")
    ax.scatter(9,irrr_l1_robust_knee2.get_rms_misfit(), label="R L1 knee2")
    ax.scatter(10,irrr_l1_robust_knee3.get_rms_misfit(), label="R L1 knee3")
    ax.set_xlabel("Solution")
    ax.set_ylabel('Root mean square misfit')
    ax.set_title('Root mean square of the solutions')
    ax.legend()
    imgs.append(Figurehelper(fig).to_png())

    return render(request, 'main/assignment_1_full.html', {
        'l1_imgs': imgs,
    })


def assignment_1_gcv(request):
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    R = L.T @ L

    point_size = 40

    ## L1 solution
    l1_imgs = []
    alphas = numpy.geomspace(0.00001, 1, num=20)  # Search from 10^-3 to 10^-0
    random_data = numpy.random.randint(len(d), size=50)

    datums = []
    d_gcvs = []
    G_r_gcvs = []
    start_time = time.time()
    for j, step in enumerate(random_data):
        datums.append(d[int(step)])
        d_gcvs.append(numpy.delete(d, int(step)))
        theta_gcv = numpy.delete(theta, int(step))
        phi_gcv = numpy.delete(phi, int(step))
        r_gcv = numpy.delete(r, int(step))
        G_r_gcv, _, _ = design_gauss(r_gcv, theta_gcv, phi_gcv, N)
        G_r_gcvs.append(G_r_gcv)
    print("Set up one-out", time.time() - start_time)

    # For the  GCV method first we train the model and then apply it to data with one-out
    models = []
    gcv = numpy.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        print('alpha', len(alphas) - i, alpha)
        start_time = time.time()
        irrr = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge()
        print("converged in", time.time() - start_time)
        w = irrr.W
        m_alpha = irrr.get_model()
        for j, step in enumerate(random_data):
            print('step', len(random_data) - j)
            datum = datums[j]
            d_gcv = d_gcvs[j]
            G_r_gcv = G_r_gcvs[j]
            start_time = time.time()
            w_gcv = numpy.delete(numpy.delete(w, j, axis=0), j, axis=1)
            irrr.W = w_gcv
            irrr.d = d_gcv
            irrr.G = G_r_gcv

            trace = numpy.trace(numpy.identity(len(d) - 1) - G_r_gcv @ irrr.get_design_alpha())
            misfit = numpy.square((G_r_gcv @ m_alpha - datum) / trace)
            print("Apply gcv", time.time() - start_time)
            gcv[i] += numpy.sum(misfit)

    gcv_min = numpy.argmin(gcv)
    alpha = alphas[gcv_min]
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(alphas, gcv)
    ax.scatter(alphas[gcv_min], gcv[gcv_min], label='alpha= ' + str(alpha), facecolor=None,
               edgecolors='r')
    ax.set_ylabel("Misfit")
    ax.set_xlabel("alpha")
    ax.set_xscale('log')
    ax.legend()
    ax.set_title("Generalized cross validation")
    l1_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(alphas, gcv)
    ax.scatter(alphas[gcv_min], gcv[gcv_min], label='alpha= ' + str(alpha), facecolor=None,
               edgecolors='r')
    ax.set_ylabel("Misfit")
    ax.set_xlabel("alpha")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title("Generalized cross validation")
    l1_imgs.append(Figurehelper(fig).to_png())

    log_alphas = numpy.log(alphas)
    fig, ax = pyplot.subplots(figsize=(12, 6))
    norm = colors.Normalize(vmin=log_alphas.min(), vmax=log_alphas.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    ax.plot(alphas, gcv)
    ax.scatter(alphas, gcv, c=numpy.log(alphas))
    cbar = fig.colorbar(cmap, ticks=numpy.log(alphas))
    cbar.ax.set_ylabel('Alpha')
    cbar.ax.set_yticklabels(alphas)
    ax.set_ylabel("Misfit")
    ax.set_xlabel("alpha")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Generalized cross validation")
    l1_imgs.append(Figurehelper(fig).to_png())

    m_ls = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge().converged_model
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Regularized robust L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    l1_imgs.append(ls2)

    return render(request, 'main/assignment_1_gcv.html', {
        'l1_imgs': l1_imgs,
    })


def assignment_1_disc(request):
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    R = L.T @ L

    point_size = 40

    ## L1 solution
    l1_imgs = []
    alphas = numpy.geomspace(0.00001, 1, num=20)
    misfit_norms = []
    sigma_rms_ls = IRRR(d, Gr, sigma=1).converge().get_rms_misfit()
    print(sigma_rms_ls)
    for i, alpha in enumerate(alphas):
        print(len(alphas) - i)
        irrr = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge()
        print(irrr.iterations, alpha, irrr.get_rms_misfit())
        print(irrr.model_norms[irrr.converged_iteration], irrr.misfit_norms[irrr.converged_iteration],
              numpy.sqrt(irrr.model_norms[irrr.converged_iteration]))
        misfit_norms.append(irrr.get_misfit_norm())

    misfit_desc = numpy.abs(numpy.asarray(misfit_norms) - len(d) * sigma_rms_ls)
    desc_index = numpy.argmin(misfit_desc)
    alpha = alphas[desc_index]

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(alphas, misfit_desc)
    ax.scatter(alphas[desc_index], misfit_desc[desc_index], label='alpha= ' + str(alpha), facecolor=None,
               edgecolors='r')
    ax.set_ylabel("Misfit")
    ax.set_xlabel("alpha")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title("Discrepancy principle")
    l1_imgs.append(Figurehelper(fig).to_png())

    m_ls = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge().converged_model
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Regularized robust L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    l1_imgs.append(ls2)

    return render(request, 'main/assignment_1_disc.html', {
        'l1_imgs': l1_imgs,
    })


def assignment_1_knee(request):
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    R = L.T @ L

    point_size = 40

    ## L1 solution
    l1_imgs = []
    alphas = numpy.geomspace(0.00000001, 0.1, num=20)  # numpy.geomspace(0.00001, 0.001, num=20) for knee curve
    model_norms = []
    misfit_norms = []
    conv_model_norms = []
    conv_misfit_norms = []
    # irrr = IRRR(d, Gr, alpha1=0.0025, L1=L, norm=1, weight_type=IRLS.HUBER).converge()
    # print(irrr.iterations, 0.002, irrr.get_rms_misfit())
    rms_misfits = []
    for i, alpha in enumerate(alphas):
        print(len(alphas) - i)
        irrr = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge()
        print(irrr.iterations, alpha, irrr.model_norms[-1], irrr.misfit_norms[-1], irrr.rms_misfits[-1])

        model_norms.append(irrr.get_model_norm())
        misfit_norms.append(irrr.get_misfit_norm())
        rms_misfits.append(irrr.rms_misfits)
        conv_misfit_norms.append(irrr.misfit_norms[irrr.converged_iteration])
        conv_model_norms.append(irrr.model_norms[irrr.converged_iteration])
    curvature = numpy.abs(numpy.gradient(numpy.gradient(model_norms))) + numpy.abs(
        numpy.gradient(numpy.gradient(misfit_norms)))
    alpha_index = numpy.argmax(curvature)
    alpha = alphas[alpha_index]
    print(alpha)
    print(numpy.log(alpha))

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(misfit_norms, model_norms)
    ax.scatter(misfit_norms[alpha_index], model_norms[alpha_index], c='r',
               label='alpha= ' + str(numpy.round(alpha, 7)))
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Knee - model norm vs misfit")
    l1_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(alphas, curvature, '-', label='Max curvature alpha= ' + str(numpy.round(alpha, 5)))
    ax.set_xscale('log')
    ax.set_xlabel("alpha")
    ax.set_ylabel("Curvature")
    ax.set_title("Curvature vs alpha")
    ax.legend()
    l1_imgs.append(Figurehelper(fig).to_png())

    log_alphas = numpy.log(alphas)
    norm = colors.Normalize(vmin=log_alphas.min(), vmax=log_alphas.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    cmap.set_array([])
    fig, ax = pyplot.subplots(figsize=(12, 6))
    for i, yi in enumerate(rms_misfits):
        ax.plot(numpy.arange(len(yi)), yi, c=cmap.to_rgba(log_alphas[i]))
    cbar = fig.colorbar(cmap, ticks=log_alphas[::2])
    cbar.ax.set_ylabel('Alpha')
    cbar.ax.set_yticklabels(alphas[::2])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Rms misfit")
    ax.set_yscale('log')
    ax.set_title("Convergence of alpha values")
    l1_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(12, 6))
    norm = colors.Normalize(vmin=log_alphas.min(), vmax=log_alphas.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    ax.scatter(conv_misfit_norms, conv_model_norms, c=numpy.log(alphas))
    cbar = fig.colorbar(cmap, ticks=numpy.log(alphas)[::10])
    cbar.ax.set_ylabel('Alpha')
    cbar.ax.set_yticklabels(alphas[::10])
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Knee - model norm vs misfit norm")
    l1_imgs.append(Figurehelper(fig).to_png())

    alpha = 0.0001  # 0.00464  # is a good value
    m_ls = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge().converged_model
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Regularized robust L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    l1_imgs.append(ls2)

    alpha = 0.00015  # 0.00464  # is a good value
    m_ls = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge().converged_model
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Regularized robust L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    l1_imgs.append(ls2)

    alpha = 0.0002  # 0.00464  # is a good value
    m_ls = IRRR(d, Gr, alpha1=alpha, L1=L, norm=1, weight_type=IRLS.HUBER).converge().converged_model
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Regularized robust L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    l1_imgs.append(ls2)

    return render(request, 'main/assignment_1_knee.html', {
        'l1_imgs': l1_imgs,
    })


def assignment_1_tikh(request):
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    R = L.T @ L

    point_size = 40

    tikh_l2_imgs = []
    alphas = numpy.geomspace(0.0000000001, 0.1,
                             num=20)  # numpy.geomspace(0.0000007, 0.1, num=20)  # numpy.geomspace(0.00000001, 0.000004, num=20) for knee l2
    model_norms = []
    misfit_norms = []
    for i, alpha in enumerate(alphas):
        print(i)
        irrr = IRRR(d, Gr, alpha1=alpha, norm=2).converge()
        print(irrr.iterations, "iterations")
        model_norms.append(irrr.get_model_norm())
        misfit_norms.append(irrr.get_misfit_norm())
    curvature = numpy.sqrt(numpy.square(numpy.gradient(numpy.gradient(model_norms))) + numpy.square(
        numpy.gradient(numpy.gradient(misfit_norms))))

    alpha_index = numpy.argmax(curvature)
    alpha = alphas[alpha_index]
    print(alpha_index)

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(misfit_norms, model_norms)
    ax.scatter(misfit_norms[alpha_index], model_norms[alpha_index], c='r',
               label='alpha= ' + str(numpy.round(alpha, 7)))
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm")
    ax.set_title("Knee - model norm vs misfit")
    ax.legend()
    tikh_l2_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(misfit_norms, model_norms)
    ax.scatter(misfit_norms[alpha_index], model_norms[alpha_index], c='r',
               label='alpha= ' + str(numpy.round(alpha, 7)))
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Knee - model norm vs misfit")
    ax.legend()
    tikh_l2_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(alphas, curvature, '-', )
    ax.scatter(alphas[alpha_index], curvature[alpha_index], c='r',
               label='Max curvature alpha= ' + str(numpy.round(alpha, 7)))
    ax.set_xscale('log')
    ax.set_xlabel("alpha")
    ax.set_ylabel("Curvature")
    ax.set_title("Curvature vs alpha")
    ax.legend()
    tikh_l2_imgs.append(Figurehelper(fig).to_png())

    m_ls = IRRR(d, Gr, alpha1=alpha, norm=2, L1=L).converge().get_model()
    Br_ls = Gr_cmb @ m_ls
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="L1 norm, CMB, nmax={}".format(N),
                      cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    tikh_l2_imgs.append(ls2)

    log_alphas = numpy.log(alphas)
    fig, ax = pyplot.subplots(figsize=(12, 6))
    norm = colors.Normalize(vmin=log_alphas.min(), vmax=log_alphas.max())
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    ax.scatter(misfit_norms, model_norms, c=numpy.log(alphas))
    cbar = fig.colorbar(cmap, ticks=numpy.log(alphas))
    cbar.ax.set_ylabel('Alpha')
    cbar.ax.set_yticklabels(alphas)
    ax.set_xlabel("Misfit norm")
    ax.set_ylabel("Model norm")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title("Knee - model norm vs misfit norm")
    tikh_l2_imgs.append(Figurehelper(fig).to_png())

    return render(request, 'main/assignment_1_tikh.html', {
        'tikh_l2_imgs': tikh_l2_imgs,
    })


def assignment_1_data(request):
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)
    print(d.shape, theta.shape, phi.shape, r.shape)
    print(theta.max(), theta.min(), phi.max(), phi.min())
    print(d.mean(), numpy.sqrt(d.var()))
    data_imgs = []

    n_bins = int(2 * len(d) ** (-1 / 3) * (
            numpy.median(d[int(len(numpy.sort(d)) / 2) + 1:None]) - numpy.median(d[0:int(len(numpy.sort(d)) / 2)])))

    print(n_bins)

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(numpy.sort(d), numpy.arange(len(d)))
    ax.set_xlabel("Data value")
    ax.set_ylabel("Count")
    ax.set_title("Data distrubution by value")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.hist(d, bins=n_bins)
    ax.set_xlabel("Data value")
    ax.set_ylabel("Count")
    ax.set_title("Data distrubution by value")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    L1 = numpy.abs(numpy.mean(d) - d)
    ax.hist(L1, bins=n_bins)
    ax.set_xlabel("L1 norm")
    ax.set_ylabel("Data count")
    ax.set_title("L1 norm mean")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    L1 = numpy.abs(numpy.median(d) - d)
    ax.hist(L1, bins=n_bins)
    ax.set_xlabel("L1 norm")
    ax.set_ylabel("Data count")
    ax.set_title("L1 norm median")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    L2 = numpy.square(numpy.mean(d) - d)
    ax.hist(L2, bins=n_bins)
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("L2 norm mean")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    L2 = numpy.square(numpy.median(d) - d)
    ax.hist(L2, bins=n_bins)
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("L2 norm median")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.hist(theta, bins=180)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Count")
    ax.set_title("Data distrubution by latitude")
    data_imgs.append(Figurehelper(fig).to_png())

    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.hist(phi, bins=360)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Count")
    ax.set_title("Data distrubution by longitude")
    data_imgs.append(Figurehelper(fig).to_png())

    counts, start, dx, _ = cumfreq(d, numbins=n_bins)
    x = numpy.arange(counts.size) * dx + start
    fig, ax = pyplot.subplots(figsize=(6, 6))
    ax.plot(x, counts / len(d), 'ro')
    ax.set_ylabel("Probability")
    ax.set_xlabel("Value")
    ax.set_title("Normal probability")
    data_imgs.append(Figurehelper(fig).to_png())

    return render(request, 'main/assignment_1_data.html', {
        'data_imgs': data_imgs,
    })


def assignment_1(request):
    # Constants needed
    a = 6371.2  # Earth radius in km
    c = 3480.0  # Core radius in km
    # Power spectra radius
    ps_r = c

    # Load Swarm data file located in current directory
    data = CourseDataLoader.load_mat("SW_B_14_28_Sept14_selected.mat")
    d = data["Br"].reshape(-1, )  # Select radial field data
    theta = data["theta"].reshape(-1, )
    phi = data["lambda"].reshape(-1, )  # lambda is already used by Python, use phi instead
    r = data["r"].reshape(-1)

    # Set SH degree of model
    N = 20

    # Design matrix for B_r at data location
    Gr, _, _ = design_gauss(r, theta, phi, N)

    # Setup Regularization Matrix
    # Gr_cmb is linear operator matrix L for this problem, produces predictions of B_r at the CMB
    # L2 norm m^T Gr_cmb^T Gr_cmb m approximates B_r^2 integrated over a grid at the CMB
    # L1 norm approximate abs(B_r) integrated over a grid at the CMB

    # phi_cmb, theta_cmb = numpy.meshgrid(numpy.linspace(-180., 180., num=361), numpy.linspace(0., 180., num=181))
    phi_cmb, theta_cmb = numpy.meshgrid(numpy.arange(-180., 180., 5),
                                        numpy.arange(0., 180. + 5, 5))  # Step size of 5, W_md: 2664x2664
    phi_cmb = phi_cmb.reshape(-1, )
    theta_cmb = theta_cmb.reshape(-1, )
    r_cmb = c * numpy.ones(theta_cmb.shape)

    Gr_cmb, _, _ = design_gauss(r_cmb, theta_cmb, phi_cmb, N)
    L = Gr_cmb
    R = L.T @ L

    #  Plot simple map of B_r data
    point_size = 10
    img1 = plot_global(phi, 90 - theta, d, point_size, title="Swarm satellite observations", cbar_label="Br [nT]",
                       cmap=pyplot.cm.PuOr_r)

    # Least square inversion
    # m_ls=numpy.linalg.lstsq(Gr, d, rcond=None)[0]
    m_ls = numpy.linalg.inv(Gr.T @ Gr) @ Gr.T @ d  # with no SVD

    Br_ls = Gr_cmb @ m_ls

    # LS CMB plot
    img2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                       title="LS solution, CMB, nmax={}".format(N), cbar_label="Br [mT]", cmap=pyplot.cm.PuOr_r)

    # Power spectrum plot

    # Power spectrum and CMB field
    ps_ls = model_utils.power_spectrum(m_ls, radius=ps_r)
    n = numpy.arange(1, N + 1)
    fig, ax = pyplot.subplots(figsize=(9, 4))
    ax.semilogy(n, ps_ls, label="LS")
    ax.set_xlabel("degree n")
    ax.set_ylabel(r"Power [nT$^2$]")
    ax.set_title(r"Power spectra of estimated geomagnetic field at $r={}$km".format(ps_r))
    nmax = numpy.max([len(n)])
    ax.set_xticks(numpy.arange(1, nmax + 1))
    ax.legend()
    ax.grid()
    img3 = Figurehelper(fig).to_png()

    # Residuals
    r_ls = d - Gr @ m_ls
    # rms misfit
    rms_resid_ls = numpy.sqrt(r_ls.T @ r_ls / len(d))
    print("rms misfit: {:.1f}nT".format(float(rms_resid_ls)))

    least_squares_imgs = []
    ## LS SOLUTION
    irrr = IRRR(d, Gr)
    m_ls = irrr.update().get_model()
    Br_ls = Gr_cmb @ m_ls
    ls1 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="LS solution, CMB, nmax={}".format(N), cbar_label="Br [mT]", cmap=pyplot.cm.PuOr_r)
    least_squares_imgs.append(ls1)
    ## Iterative least squares huber weights
    irrr = IRRR(d, Gr, weight_type=IRLS.HUBER, truncate=False, max_iterations=15).converge()
    print(irrr.iterations)
    m_ls_w = irrr.get_model()
    Br_ls = Gr_cmb @ m_ls_w
    ls2 = plot_global(phi_cmb, 90 - theta_cmb, Br_ls * 10 ** (-6), point_size,
                      title="Weighted LS solution huber weights, CMB, nmax={}".format(N), cbar_label="Br [mT]",
                      cmap=pyplot.cm.PuOr_r)
    least_squares_imgs.append(ls2)

    ps_ls = model_utils.power_spectrum(m_ls, radius=ps_r)
    ps_ls_w = model_utils.power_spectrum(m_ls_w, radius=ps_r)
    n = numpy.arange(1, N + 1)
    fig, ax = pyplot.subplots(figsize=(9, 4))
    ax.semilogy(n, ps_ls, label="LS")
    ax.semilogy(n, ps_ls_w, label="weighted LS")
    ax.set_xlabel("degree n")
    ax.set_ylabel(r"Power [nT$^2$]")
    ax.set_title(r"Power spectra of estimated geomagnetic field at $r={}$km".format(ps_r))
    nmax = numpy.max([len(n)])
    ax.set_xticks(numpy.arange(1, nmax + 1))
    ax.legend()
    ax.grid()
    img3 = Figurehelper(fig).to_png()

    return render(request, 'main/assignment_1.html', {
        'ls_imgs': least_squares_imgs,
        'img1': img1,
        'img2': img2,
        'img3': img3,
    })


def exercise_3_4(request):
    def G_crust(x_j, x_m, h):
        # Regarding constant pre-factor [not crucial for understanding]
        # -\\mu_0/2pi = -2 x10-^7
        # But Magnetization in SI Am^-1, data in nT=10^-9 T
        # So finally correct prefactor is -200 [See Parker, 1994 for details]
        x_m = x_m.reshape(1, -1)
        x_j = x_j.reshape(-1, 1)
        g = -200 * ((x_m - x_j) ** 2 - h ** 2) / ((x_m - x_j) ** 2 + h ** 2) ** 2
        return g

    delta_x_m = 0.1
    x_m = numpy.arange(-25, 25 + delta_x_m, delta_x_m)
    # define data locations
    x_j = numpy.arange(-15, 15 + 1, 1)
    # Build design matrix
    h = 2  # Depth of water (km)
    G = G_crust(x_j, x_m, h) * delta_x_m

    # Define synthetic model
    m = numpy.zeros((len(x_m), 1))
    ones = numpy.ones((51, 1))

    m[numpy.arange(250 - 127, 251 - 77)] = 2 * ones
    m[numpy.arange(250 - 76, 251 - 26)] = -2 * ones
    m[numpy.arange(250 - 25, 251 + 25)] = 2 * ones
    m[numpy.arange(250 + 26, 251 + 76)] = -2 * ones
    m[numpy.arange(250 + 77, 251 + 127)] = 2 * ones
    # Produce synthetic data
    d_j = G @ m

    # Add Gaussian noise of amplitude 8 nT
    dn_j = 8 * numpy.random.rand(len(d_j), 1)
    d_j = d_j + dn_j

    # Add Outliers
    # d_j[12]=300
    # d_j[20]=d_j[19]

    # Plot data
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(x_j, d_j, 'ko')
    ax.set_xlabel('Distance from ridge axis (km)')
    ax.set_ylabel('Magnetic anomaly (nT)')
    img1 = Figurehelper(fig).to_png()

    # Simple least squares soln.
    m_ls = numpy.linalg.lstsq(G, d_j, rcond=None)[0]  # Python lstsq already included regularization ?

    # Plot simple least squares soln
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.plot(x_m, m_ls, '-r', label='least squares')
    ax.set_xlabel('Distance from ridge axis (km)')
    ax.set_ylabel(r'Magnetization (Am$^{-1}$)')

    # Residuals
    r_ls = d_j - G @ m_ls
    # rms misfit
    rms_resid_ls = numpy.sqrt(r_ls.T @ r_ls / len(d_j))

    # L2 norm Tikhonov Regularized solution
    alpha_sq = 3.95E1  # For discrepancy principle
    GTG = G.T @ G

    # Determine Tikhonov solution
    # m_L2tik = np.linalg.lsq(GtG+alpha_sq*eye(length(x_m),length(x_m)))\\G'*d_j;
    I = numpy.identity(len(GTG))
    m_L2tik = numpy.linalg.lstsq((GTG + alpha_sq * I), G.T @ d_j, rcond=None)[0]

    # Residuals
    r_L2tik = d_j - G @ m_L2tik
    # rms misfit
    rms_resid_L2tik = numpy.sqrt(r_L2tik.T @ r_L2tik / len(d_j))

    ## Plot regularized solution: much more reasonable
    ax.plot(x_m, m_L2tik, '-b', label='Reg tikh L2')
    ax.plot(x_m, m, 'k', label='Expected')
    ax.legend()
    img2 = Figurehelper(fig).to_png()

    L2 = numpy.zeros((len(G.T), len(G.T)))
    for i in range(len(G.T)):
        for j in range(len(G.T)):
            if i == j:
                L2[i, j] = -1
                if j != len(G.T) - 1:
                    L2[i, j + 1] = 1
            if i == len(G.T) - 1 and j == len(G.T) - 1:
                L2[i, j] = 1
                L2[i, j - 1] = -1
    n_sigma = len(d_j) * 8
    n = 1

    alpha1_l2 = 4.27
    alpha1_l1 = 15.23

    alphas = numpy.geomspace(1, 16, num=5)
    misfit_1 = numpy.zeros(len(alphas))
    misfit_2 = numpy.zeros(len(alphas))
    for i in range(len(alphas)):
        print(n - i)
        irls_l2 = IRRR(d_j, G, norm=2, L1=I, alpha1=alphas[i]).update()
        misfit_1[i] = numpy.square(irls_l2.get_misfit_norm() - n_sigma)
        irls_l1 = IRRR(d_j, G, L1=I, norm=1, alpha1=alphas[i]).update()
        misfit_2[i] = numpy.square(irls_l1.get_misfit_norm() - n_sigma)

    print(alphas[numpy.argmin(misfit_1)])
    print(alphas[numpy.argmin(misfit_2)])
    alphas_total = numpy.geomspace(0.001, 1, num=n)
    misfit_3 = numpy.zeros(len(alphas_total))
    misfit_4 = numpy.zeros(len(alphas_total))
    for i in range(len(alphas_total)):
        print(n - i)
        irls_l1_total = IRRR(d_j, G, L1=I, L2=L2, norm=1, alpha1=alpha1_l1, alpha2=alphas_total[i]).converge()
        misfit_3[i] = numpy.square(irls_l1_total.get_misfit_norm() - n_sigma)
        irls_l1_totalt_huber = IRRR(d_j, G, L1=I, L2=L2, type=IRLS.HUBER, norm=1, alpha1=alpha1_l1,
                                    alpha2=alphas_total[i]).converge()
        misfit_4[i] = numpy.square(irls_l1_totalt_huber.get_misfit_norm() - n_sigma)
    print(alphas_total[numpy.argmin(misfit_3)])
    print(alphas_total[numpy.argmin(misfit_4)])
    alpha2_l1_ls = 0.46
    alpha2_l1_huber = 0.46

    alpha_figs = []
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.set_xlabel('alpha')
    ax.set_ylabel('misfit')
    ax.plot(alphas, misfit_1, '-b', label='L2 norm')
    ax.plot(alphas, misfit_2, '-r', label='L1 norm')
    ax.legend()
    alpha_figs.append(Figurehelper(fig).to_png())
    fig, ax = pyplot.subplots(figsize=(12, 6))
    ax.set_xlabel('alpha')
    ax.set_ylabel('misfit')
    ax.plot(alphas_total, misfit_3, '-g', label='L1 norm total')
    ax.plot(alphas_total, misfit_4, '-y', label='L1 norm total huber')
    ax.legend()
    alpha_figs.append(Figurehelper(fig).to_png())

    alpha_1 = 11  # alphas[numpy.argmin(misfit_1)]
    alpha_2 = 11  # alphas[numpy.argmin(misfit_2)]

    irrr_1 = IRRR(d_j, G, I, alpha1=alpha1_l1, norm=1).converge()
    m_irrr_1 = irrr_1.get_model()
    irrr_2 = IRRR(d_j, G, I, norm=2, alpha1=alpha1_l2).converge()
    m_irrr_2 = irrr_2.get_model()

    irrr_3 = IRRR(d_j, G, I, L2=L2, alpha1=alpha1_l1, alpha2=alpha2_l1_ls).converge()
    m_irrr_3 = irrr_3.get_model()

    irrr_4 = IRRR(d_j, G, I, L2=L2, alpha1=alpha1_l1, alpha2=alpha2_l1_huber, type=IRLS.HUBER).converge()
    m_irrr_4 = irrr_4.get_model()

    d_j_outlier = d_j.copy()
    d_j_outlier[15] = 2 * numpy.max(d_j)
    d_j_outlier[5] = 2 * numpy.min(d_j)

    irrr_5 = IRRR(d_j_outlier, G, I, L2=L2, alpha1=alpha1_l1, alpha2=alpha2_l1_huber, type=IRLS.HUBER).converge()
    m_irrr_5 = irrr_5.get_model()

    fig, ax = pyplot.subplots(figsize=(24, 12))
    ax.set_xlabel('Distance from ridge axis (km)')
    ax.set_ylabel(r'Magnetization (Am$^{-1}$)')
    ax.plot(x_m, m_irrr_1, '-b', label='L1 norm, alpha= ' + str(numpy.round(alpha1_l1, 2)))
    ax.plot(x_m, m_irrr_2, '-r', label='L2 norm, alpha= ' + str(numpy.round(alpha1_l2, 2)))
    ax.plot(x_m, m_irrr_3, '-g', label='L1 total var, alpha= ' + str(numpy.round(alpha2_l1_ls, 2)))
    ax.plot(x_m, m_irrr_4, '-y', label='L1 total var huber, alpha= ' + str(numpy.round(alpha2_l1_huber, 2)))
    ax.plot(x_m, m_irrr_5, '-m', label='L1 total var outlier huber, alpha= ' + str(numpy.round(alpha2_l1_huber, 2)))
    ax.plot(x_m, m, 'k', label='Expected')
    ax.set_ylim(-3, 3)
    ax.legend()
    img3 = Figurehelper(fig).to_png()

    return render(request, 'main/exercise_3_4.html', {
        'img1': img1,
        'img2': img2,
        'img3': img3,
        'alpha_figs': alpha_figs,
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
