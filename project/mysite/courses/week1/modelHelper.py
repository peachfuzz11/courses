from matplotlib import pyplot

from courses.week1.irls import IRLS
from courses.week2.robust import Robust
from main.helpers.figurehelper import Figurehelper


class ModelHelper:

    @staticmethod
    def get_robust_plot(model: Robust) -> {}:
        models = model.get_model_vectors()
        a1, a2, a3 = zip(*[(m[0], m[1], m[2]) for m in models])
        figs = []
        p = 0
        for a in [a1, a2, a3]:
            p = p + 1
            i = model.iterations
            fig1, ax1 = pyplot.subplots(figsize=(6, 6))
            ax1.scatter(range(i), a, facecolors='none', edgecolors='red')
            ax1.set_title('param:' + str(p) + ' converges to ' + str(a[-1]) + ' after ' + str(i) + ' iterations')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('parameter value')
            img1 = Figurehelper(fig1).to_png()
            figs.append(img1)
        return figs

    @staticmethod
    def getAttribute(model: IRLS) -> {}:
        iterations = 15
        model.iterate(iterations)
        mean = model.get_mean()[-1]

        fig1, ax1 = pyplot.subplots(figsize=(6, 6))
        ax1.set_title('Mean:' + str(mean) + ' after ' + str(iterations) + ' iterations')
        ax1.scatter(range(iterations), model.get_mean(), facecolors='none', edgecolors='red')
        ax1.set_xlabel('iterations')
        ax1.set_ylabel('mean')
        img1 = Figurehelper(fig1).to_png()

        if True:
            model.reset()
            percent = 1
            i = model.percent_of_median(percent)

            fig2, ax2 = pyplot.subplots(figsize=(6, 6))
            ax2.set_title('Iterations:' + str(i) + ' to reach ' + str(percent) + ' percent of median')
            ax2.scatter(range(i), model.get_mean(), facecolors='none', edgecolors='red')
            ax2.set_xlabel('iterations')
            ax2.set_ylabel('mean')
            img2 = Figurehelper(fig2).to_png()

        return {
            'name': model.type,
            'mean': mean,
            'fig1': img1,
            'fig2': img2,
            'iterations': iterations,
        }
