# coding=utf-8
import math
from matplotlib.ticker import FuncFormatter


def f_measure_pdf(rand_xs, a, b):
    import scipy.special as ss
    pdf_values = []
    for rand_x in rand_xs:
        nom_ln = a * math.log(2) + (a - 1) * math.log(1 - rand_x) + (-a - b) * math.log(2 - rand_x) + (b - 1) * math.log(rand_x)
        den_ln = ss.betaln(a, b)
        ln_val = nom_ln - den_ln
        pdf_values.append(math.exp(ln_val))
    return pdf_values
    pass


def mode_f_measure(a, b):
    import math
    t = -0.5 * b - 0.25 * a + 1.25 + 0.25 * (math.sqrt(4 * b * b + 4 * a * b - 4 * b + a * a - 10 * a + 9))
    return [t, f_measure_pdf(t, a, b)]
    pass


def f_measure_ppf(q, a, b):
    from scipy.stats import betaprime
    betaprime_ppf = betaprime.ppf(1 - q, a, b)
    f_ppf = 1.0 / (1 + 0.5 * betaprime_ppf)
    return f_ppf


def to_percent(temp, position):
    return '%1.0f'%(10*temp) + '%'


def plot_pdf_line(ppf_func, pdf_func, a, b, ax, lty="r-", lw=2, legend_label=""):
    import numpy as np
    x = np.linspace(ppf_func(0.001, a, b), ppf_func(0.999, a, b), 100)
    ax.plot(x, pdf_func(x, a, b), lty, lw=lw, alpha=0.6, label=legend_label)
    for xtick in ax.get_xticklabels():
        xtick.set_rotation(20)
    ax.get_xaxis().set_minor_formatter(FuncFormatter(to_percent))
    pass


def compute_aggrate_conf_matrix_mx2(results, rho_avg=0.3688):
    result_mx2 = [0, 0, 0]
    assert len(results) % 2 == 0
    for result in results:
        tp = result[2]
        fn = result[0] - result[2]
        fp = result[1] - result[2]
        result_mx2[0] += tp
        result_mx2[1] += fn
        result_mx2[2] += fp
    tp_mx2 = result_mx2[0] * rho_avg
    fn_mx2 = result_mx2[1] * rho_avg
    fp_mx2 = result_mx2[2] * rho_avg
    return tp_mx2, fn_mx2, fp_mx2
    pass


def demo_plot_R(results_bmes, ax, color='r', labels =["","",""]):
    from scipy.stats import beta
    tp_bmes_5x2, fn_bmes_5x2, fp_bmes_5x2 = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=0.3688)
    plot_pdf_line(beta.ppf, beta.pdf, tp_bmes_5x2 + 1, fn_bmes_5x2 + 1, ax, lty=color + '-', lw=2, legend_label=labels[0])
    tp_bmes_5x2_cor, fn_bmes_5x2_cor, fp_bmes_5x2_cor = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=1/6.0)
    plot_pdf_line(beta.ppf, beta.pdf, tp_bmes_5x2_cor + 1, fn_bmes_5x2_cor + 1, ax, lty=color + '-.', lw=1, legend_label=labels[1])
    pass


def demo_plot_P(results_bmes, ax, color='r', labels =["","",""]):
    from scipy.stats import beta
    tp_bmes_5x2, fn_bmes_5x2, fp_bmes_5x2 = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=0.3688)
    plot_pdf_line(beta.ppf, beta.pdf, tp_bmes_5x2 + 1, fp_bmes_5x2 + 1, ax, lty=color + '-', lw=2, legend_label=labels[0])
    tp_bmes_5x2_cor, fn_bmes_5x2_cor, fp_bmes_5x2_cor = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=1/6.0)
    plot_pdf_line(beta.ppf, beta.pdf, tp_bmes_5x2_cor + 1, fp_bmes_5x2_cor + 1, ax, lty=color + '-.', lw=1, legend_label=labels[1])
    pass


def demo_plot_F1(results_bmes, ax, color='r',labels =["","",""]):
    tp_bmes_5x2, fn_bmes_5x2, fp_bmes_5x2 = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=0.3688)
    plot_pdf_line(f_measure_ppf, f_measure_pdf, fp_bmes_5x2 + fn_bmes_5x2 + 2, tp_bmes_5x2 + 1, ax, lty=color + '-',
                  lw=2, legend_label=labels[0])
    tp_bmes_5x2_cor, fn_bmes_5x2_cor, fp_bmes_5x2_cor = compute_aggrate_conf_matrix_mx2(results_bmes, rho_avg=1/6.0)
    plot_pdf_line(f_measure_ppf, f_measure_pdf, fp_bmes_5x2_cor + fn_bmes_5x2_cor + 2, tp_bmes_5x2_cor + 1, ax,
                  lty=color + '-.', lw=1, legend_label=labels[1])
    pass
