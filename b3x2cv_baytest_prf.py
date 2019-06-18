# coding=utf-8
from enum import Enum

class F1MeasureDist:

    def __init__(self, tp, fp, fn, lambd = 1):
        self.a = fp + fn + 2 * lambd
        self.b = tp + lambd
        self.tp = tp
        self.fp = fp
        self.fn = fn
        pass

    def ppf(self, q):
        from scipy.stats import betaprime
        beta_prime_ppf = betaprime.ppf(1 - q, self.a, self.b)
        f_ppf = 1.0 / (1 + 0.5 * beta_prime_ppf)
        return f_ppf

    def pdf(self, rand_xs):
        a = self.a
        b = self.b
        import math
        import scipy.special as ss
        pdf_values = None
        if isinstance(rand_xs, float) or isinstance(rand_xs, int):
            rand_x = 1.0 * rand_xs
            nom_ln = a * math.log(2) + (a - 1) * math.log(1 - rand_x) + (-a - b) * math.log(2 - rand_x) + (b - 1) * math.log(rand_x)
            den_ln = ss.betaln(a, b)
            ln_val = nom_ln - den_ln
            pdf_values = math.exp(ln_val)
            pass
        elif isinstance(rand_xs, list):
            pdf_values = []
            for rand_x in rand_xs:
                nom_ln = a * math.log(2) + (a - 1) * math.log(1 - rand_x) + (-a - b) * math.log(2 - rand_x) + (
                            b - 1) * math.log(rand_x)
                den_ln = ss.betaln(a, b)
                ln_val = nom_ln - den_ln
                pdf_values.append(math.exp(ln_val))
        return pdf_values

    def mode(self):
        import math
        a = self.a
        b = self.b
        t = -0.5 * b - 0.25 * a + 1.25 + 0.25 * (math.sqrt(4 * b * b + 4 * a * b - 4 * b + a * a - 10 * a + 9))
        return [t, self.pdf([t])[0]]
        pass

    def rvs(self, size=1000):
        a = self.a
        b = self.b
        from scipy.stats import betaprime
        beta_prime_rvs = betaprime.rvs(a, b, size=size)
        return [1.0 / (1 + 0.5 * rv) for rv in beta_prime_rvs]
        pass

    pass


def compute_confusion_matrix_by_results(results):
    confusion_matrices = []
    for result in results:
        tp = result[2]
        fn = result[0] - result[2]
        fp = result[1] - result[2]
        confusion_matrices.append([tp, fn, fp])
        pass
    return confusion_matrices
    pass


def compute_effective_confusion_matrix_3x2bcv(confusion_matrices, weight=0.3688):
    effective_confusion_matrix = [0, 0, 0]
    assert len(confusion_matrices) % 2 == 0
    for confusion_matrix in confusion_matrices:
        tp = confusion_matrix[0]
        fn = confusion_matrix[1]
        fp = confusion_matrix[2]
        effective_confusion_matrix[0] += tp
        effective_confusion_matrix[1] += fn
        effective_confusion_matrix[2] += fp
    tp_3x2 = effective_confusion_matrix[0] * weight
    fn_3x2 = effective_confusion_matrix[1] * weight
    fp_3x2 = effective_confusion_matrix[2] * weight
    return tp_3x2, fn_3x2, fp_3x2
    pass


def monte_carlo(samples_a, samples_b, delta, mm_sim_count):
    diffs = [samples_b[i] - samples_a[i] for i in range(0, mm_sim_count)]
    count = 0
    for diff in diffs:
        if diff < delta:
            count += 1
    emp_prop = count * 1.0 / mm_sim_count
    return emp_prop


class Decision(Enum):
    h0_hold = 0
    h1_hold = 1


def b3x2cv_bayesian_test(confusion_matrices_a, confusion_matrices_b, lamb=1, nu="R",
                         delta=0.0, mm_sim_count=10000):
    tp_bar_a, fn_bar_a, fp_bar_a = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_a)
    tp_bar_b, fn_bar_b, fp_bar_b = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_b)
    from scipy.stats import beta
    if nu == "P":
        samples_a = list(beta.rvs(tp_bar_a + lamb, fp_bar_a + lamb, size=mm_sim_count))
        samples_b = list(beta.rvs(tp_bar_b + lamb, fp_bar_b + lamb, size=mm_sim_count))
        pass
    elif nu == "R":
        samples_a = list(beta.rvs(tp_bar_a + lamb, fn_bar_a + lamb, size=mm_sim_count))
        samples_b = list(beta.rvs(tp_bar_b + lamb, fn_bar_b + lamb, size=mm_sim_count))
        pass
    elif nu == "F1":
        samples_a = F1MeasureDist(tp_bar_a, fp_bar_a, fn_bar_a).rvs(size=mm_sim_count)
        samples_b = F1MeasureDist(tp_bar_b, fp_bar_b, fn_bar_b).rvs(size=mm_sim_count)
        pass
    else:
        raise Exception("Unsupported performance metric.")
    prob_h0 = monte_carlo(samples_a, samples_b, delta, mm_sim_count)
    prob_h1 = 1 - monte_carlo(samples_a, samples_b, delta, mm_sim_count)
    if prob_h0 >= prob_h1:
        return [Decision.h0_hold, prob_h0, prob_h1]
    else:
        return [Decision.h1_hold, prob_h0, prob_h1]
    pass


def compute_credible_interval(effective_confusion_matrix, nu = "P", lambd=1, alpha=0.05):
    from scipy.stats import beta
    import scipy.special as ss
    tp = effective_confusion_matrix[0]
    fn = effective_confusion_matrix[1]
    fp = effective_confusion_matrix[2]
    assert alpha < 0.5
    if nu == "P":
        return [beta.ppf(alpha/2, tp+lambd, fp+lambd),beta.ppf(1-alpha/2, tp+lambd, fp+lambd)]
        pass
    elif nu=="R":
        return [beta.ppf(alpha / 2, tp + lambd, fn + lambd), beta.ppf(1 - alpha / 2, tp + lambd, fn + lambd)]
        pass
    elif nu=="F1":
        f1_measure = F1MeasureDist(tp, fp, fn)
        return [f1_measure.ppf(alpha/2), f1_measure.ppf(1-alpha/2)]
        pass
    else:
        raise Exception("Unsupported performance metric")
    pass


def compute_P_R_F1_3x2(confusion_matrices):
    confusion_matrix_3x2 = [0, 0, 0]
    holdouts_prf = []
    assert len(confusion_matrices) % 2 == 0
    for confusion_matrix in confusion_matrices:
        tp = confusion_matrix[0]
        fn = confusion_matrix[1]
        fp = confusion_matrix[2]
        holdout_p = tp*1.0/(tp+fp)
        holdout_r = tp*1.0/(tp+fn)
        holdout_f1 = 2*holdout_p*holdout_r/(holdout_p+holdout_r)
        holdouts_prf.append([holdout_p, holdout_r, holdout_f1])
        confusion_matrix_3x2[0] += tp
        confusion_matrix_3x2[1] += fn
        confusion_matrix_3x2[2] += fp
    tp_3x2 = confusion_matrix_3x2[0]
    fn_3x2 = confusion_matrix_3x2[1]
    fp_3x2 = confusion_matrix_3x2[2]
    measure_P = tp_3x2*1.0/(tp_3x2+fp_3x2)
    measure_R = tp_3x2*1.0/(tp_3x2+fn_3x2)
    measure_F1 = 2*measure_P*measure_R/(measure_P+measure_R)
    return measure_P, measure_R, measure_F1, holdouts_prf
    pass


def print_confusion_matrix(confusion_matrix, label):
    print(label, "(TP", str(confusion_matrix[0])+")", "(FN", str(confusion_matrix[1])+")", "(FP", str(confusion_matrix[2])+")", sep="\t")
    pass


def print_decision_structure(decision, label):
    print(label, "["+str(decision[0]).replace('.', '\t')+"]", "[P(H0)", str(decision[1])+"]", "[P(H1)", str(decision[2])+"]" )
    pass


if __name__ == "__main__":
    """
                Test result for Bakeoff 2005 word seg task on PKU training corpus.
    """
    print("===================================================")
    print("Test decisions for Bakeoff 2005 CWS task on PKU data set")
    print("---------------------------------------------------")
    results_bmes = [[560914, 557693, 533644], [549033, 546259, 521973],
                    [555819, 553001, 528014], [554128, 551283, 527333],
                    [552806, 550079, 525858], [557141, 553922, 529186]]
    results_bb2b3mes = [[560914, 558138, 534649], [549033, 546514, 522380],
                        [555819, 553318, 528709], [554128, 551612, 527878],
                        [552806, 550544, 526632], [557141, 554312, 529580]]
    # compute confusion matrix and effective confusion matrix
    confusion_matrices_bmes = compute_confusion_matrix_by_results(results_bmes)
    confusion_matrices_bb2b3mes = compute_confusion_matrix_by_results(results_bb2b3mes)
    effective_confusion_matrix_bmes =  compute_effective_confusion_matrix_3x2bcv(confusion_matrices_bmes)
    effective_confusion_matrix_bb2b3mes = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_bb2b3mes)
    print_confusion_matrix(effective_confusion_matrix_bmes, label="Confusion matrix for BMES:")
    print_confusion_matrix(effective_confusion_matrix_bb2b3mes, label="Confusion matrix for BB2B3MES:")
    # compute credible intervals
    ci_p_bmes = compute_credible_interval(effective_confusion_matrix_bmes, nu="P")
    ci_p_bb2b3mes = compute_credible_interval(effective_confusion_matrix_bb2b3mes, nu="P")
    ci_r_bmes = compute_credible_interval(effective_confusion_matrix_bmes, nu="R")
    ci_r_bb2b3mes = compute_credible_interval(effective_confusion_matrix_bb2b3mes, nu="R")
    ci_f1_bmes = compute_credible_interval(effective_confusion_matrix_bmes, nu="F1")
    ci_f1_bb2b3mes = compute_credible_interval(effective_confusion_matrix_bb2b3mes, nu="F1")
    print("CI of P:\tBMES:"+str(ci_p_bmes)+"\tBB2B3MES:"+str(ci_p_bb2b3mes))
    print("CI of R:\tBMES:" + str(ci_r_bmes) + "\tBB2B3MES:" + str(ci_r_bb2b3mes))
    print("CI of F1:\tBMES:" + str(ci_f1_bmes) + "\tBB2B3MES:" + str(ci_f1_bb2b3mes))
    # perform Bayesian test
    decision_p = b3x2cv_bayesian_test(confusion_matrices_bmes, confusion_matrices_bb2b3mes, 1, "P",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_p, "P:")
    decision_r = b3x2cv_bayesian_test(confusion_matrices_bmes, confusion_matrices_bb2b3mes, 1, "R",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_r, "R:")
    decision_f1 = b3x2cv_bayesian_test(confusion_matrices_bmes, confusion_matrices_bb2b3mes, 1, "F1",
                                       delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_f1, "F1:")
    """
            Test result for CoNLL 2003 NER task (all entities).
    """
    print("===================================================")
    print("Test decisions for CoNLL 2003 NER task (all entities)")
    print("---------------------------------------------------")
    results_iob2 = [[11629, 11322, 10297], [11870, 11428, 10423],
                    [11718, 11370, 10333], [11781, 11402, 10347],
                    [11684, 11264, 10276], [11815, 11494, 10424]]
    results_iobes = [[11629, 11329, 10326], [11870, 11440, 10429],
                     [11718, 11357, 10354], [11781, 11386, 10339],
                     [11684, 11269, 10270], [11815, 11481, 10447]]
    # compute confusion matrix and effective confusion matrix
    confusion_matrices_iob2 = compute_confusion_matrix_by_results(results_iob2)
    confusion_matrices_iobes = compute_confusion_matrix_by_results(results_iobes)
    effective_confusion_matrix_iob2 = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_iob2)
    effective_confusion_matrix_iobes = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_iobes)
    print_confusion_matrix(effective_confusion_matrix_iob2, label="Confusion matrix for IOB2:")
    print_confusion_matrix(effective_confusion_matrix_iobes, label="Confusion matrix for IOBES:")
    # compute credible intervals
    ci_p_iob2 = compute_credible_interval(effective_confusion_matrix_iob2, nu="P")
    ci_p_iobes = compute_credible_interval(effective_confusion_matrix_iobes, nu="P")
    ci_r_iob2 = compute_credible_interval(effective_confusion_matrix_iob2, nu="R")
    ci_r_iobes = compute_credible_interval(effective_confusion_matrix_iobes, nu="R")
    ci_f1_iob2 = compute_credible_interval(effective_confusion_matrix_iob2, nu="F1")
    ci_f1_iobes = compute_credible_interval(effective_confusion_matrix_iobes, nu="F1")
    print("CI of P:\tIOB2:" + str(ci_p_iob2) + "\tIOBES:" + str(ci_p_iobes))
    print("CI of R:\tIOB2:" + str(ci_r_iob2) + "\tIOBES:" + str(ci_r_iobes))
    print("CI of F1:\tIOB2:" + str(ci_f1_iob2) + "\tIOBES:" + str(ci_f1_iobes))
    # perform Bayesian test
    decision_p = b3x2cv_bayesian_test(confusion_matrices_iob2, confusion_matrices_iobes, 1, "P",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_p, "P:")
    decision_r = b3x2cv_bayesian_test(confusion_matrices_iob2, confusion_matrices_iobes, 1, "R",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_r, "R:")
    decision_f1 = b3x2cv_bayesian_test(confusion_matrices_iob2, confusion_matrices_iobes, 1, "F1",
                                       delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_f1, "F1:")
    """
                Test result for CoNLL 2003 NER task (only ORG entities).
    """
    print("===================================================")
    print("Test decisions for CoNLL 2003 NER task (only ORG entities)")
    print("---------------------------------------------------")
    results_iob2_org = [[3146, 2242, 2088], [3175, 2298, 2109],
                        [3142, 2288, 2098], [3179, 2250, 2085],
                        [3143, 2261, 2069], [3178, 2245, 2068]]
    results_iobes_org = [[3146, 2226, 2075], [3175, 2289, 2114],
                         [3142, 2255, 2080], [3179, 2229, 2071],
                         [3143, 2210, 2041], [3178, 2218, 2054]]
    # compute confusion matrix and effective confusion matrix
    confusion_matrices_iob2_org = compute_confusion_matrix_by_results(results_iob2_org)
    confusion_matrices_iobes_org = compute_confusion_matrix_by_results(results_iobes_org)
    effective_confusion_matrix_iob2_org = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_iob2_org)
    effective_confusion_matrix_iobes_org = compute_effective_confusion_matrix_3x2bcv(confusion_matrices_iobes_org)
    print_confusion_matrix(effective_confusion_matrix_iob2_org, label="Confusion matrix for IOB2:")
    print_confusion_matrix(effective_confusion_matrix_iobes_org, label="Confusion matrix for IOBES:")
    # compute credible intervals
    ci_p_iob2_org = compute_credible_interval(effective_confusion_matrix_iob2_org, nu="P")
    ci_p_iobes_org = compute_credible_interval(effective_confusion_matrix_iobes_org, nu="P")
    ci_r_iob2_org = compute_credible_interval(effective_confusion_matrix_iob2_org, nu="R")
    ci_r_iobes_org = compute_credible_interval(effective_confusion_matrix_iobes_org, nu="R")
    ci_f1_iob2_org = compute_credible_interval(effective_confusion_matrix_iob2_org, nu="F1")
    ci_f1_iobes_org = compute_credible_interval(effective_confusion_matrix_iobes_org, nu="F1")
    print("CI of P:\tIOB2:" + str(ci_p_iob2_org) + "\tIOBES:" + str(ci_p_iobes_org))
    print("CI of R:\tIOB2:" + str(ci_r_iob2_org) + "\tIOBES:" + str(ci_r_iobes_org))
    print("CI of F1:\tIOB2:" + str(ci_f1_iob2_org) + "\tIOBES:" + str(ci_f1_iobes_org))
    # perform Bayesian test
    decision_p = b3x2cv_bayesian_test(confusion_matrices_iob2_org, confusion_matrices_iobes_org, 1, "P",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_p, "P:")
    decision_r = b3x2cv_bayesian_test(confusion_matrices_iob2_org, confusion_matrices_iobes_org, 1, "R",
                                      delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_r, "R:")
    decision_f1 = b3x2cv_bayesian_test(confusion_matrices_iob2_org, confusion_matrices_iobes_org, 1, "F1",
                                       delta=0.0, mm_sim_count=1000000)
    print_decision_structure(decision_f1, "F1:")
    pass

