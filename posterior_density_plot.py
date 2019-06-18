# coding=utf-8

from util import demo_plot_R, demo_plot_P, demo_plot_F1


def word_seg_task_beta_dist_plots():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)
    results_bmes = [[560914,557693,533644],[549033,546259,521973],[555819,553001,528014],[554128,551283,527333],
                    [552806,550079,525858],[557141,553922,529186]]
    results_bb2b3mes = [[560914, 558138, 534649], [549033, 546514, 522380], [555819, 553318, 528709],
                        [554128, 551612, 527878], [552806, 550544, 526632], [557141, 554312, 529580]]
    demo_plot_P(results_bmes, ax[0][0], color='r')
    demo_plot_P(results_bb2b3mes, ax[0][0], color='b')
    demo_plot_R(results_bmes, ax[0][1], color='r')
    demo_plot_R(results_bb2b3mes, ax[0][1], color='b')
    demo_plot_F1(results_bmes, ax[1][0], color='r', labels=["BMES (this study)", "BMES (Wang et al., 2015)"])
    demo_plot_F1(results_bb2b3mes, ax[1][0], color='b', labels=["BB2B3MES (this study)", "BB2B3MES (Wang et al., 2015)"])
    ax[0][0].set_title("P posterior")
    ax[0][1].set_title("R posterior")
    ax[1][0].set_title("F1 posterior")
    ax[0][0].tick_params(labelsize=12)
    ax[0][1].tick_params(labelsize=12)
    ax[1][0].tick_params(labelsize=12)
    ax[1][1].axis('off')
    ax[1][0].legend(loc='center', frameon=False,  numpoints=1, bbox_to_anchor=[1.67,0.5])
    leg = ax[1][0].get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)
    plt.show()
    pass


def org_ner_task_beta_dist_plots():
    import matplotlib.pyplot as plt
    results_IOB2 = [[3146,2242,2088],[3175,2298,2109],[3142,2288,2098],
                    [3179,2250,2085],[3143,2261,2069],[3178,2245,2068]]
    results_IOBES = [[3146,2226,2075],[3175,2289,2114],[3142,2255,2080],
                     [3179,2229,2071],[3143,2210,2041],[3178,2218,2054]]
    fig, ax = plt.subplots(2, 2)
    demo_plot_P(results_IOB2, ax[0][0], color='r')
    demo_plot_P(results_IOBES, ax[0][0], color='b')
    demo_plot_R(results_IOB2, ax[0][1], color='r')
    demo_plot_R(results_IOBES, ax[0][1], color='b')
    demo_plot_F1(results_IOB2, ax[1][0], color='r', labels=["IOB2 (this study)", "IOB2 (Wang et. al., 2015)"])
    demo_plot_F1(results_IOBES, ax[1][0], color='b', labels=["IOBES (this study)", "IOBES (Wang et. al., 2015)"])
    ax[0][0].set_title("P posterior")
    ax[0][1].set_title("R posterior")
    ax[1][0].set_title("F1 posterior")
    ax[0][0].tick_params(labelsize=12)
    ax[0][1].tick_params(labelsize=12)
    ax[1][0].tick_params(labelsize=12)
    ax[1][1].axis('off')
    ax[1][0].legend(loc='center', frameon=False, numpoints=1, bbox_to_anchor=[1.6, 0.5])
    leg = ax[1][0].get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)
    plt.show()
    pass


def ner_task_beta_dist_plots():
    import matplotlib.pyplot as plt
    results_IOB2 = [[11629,11322,10297],[11870,11428,10423],[11718,11370,10333],
                    [11781,11402,10347],[11684,11264,10276],[11815,11494,10424]]
    results_IOBES = [[11629,11329,10326],[11870,11440,10429],[11718,11357,10354],
                     [11781,11386,10339],[11684,11269,10270],[11815,11481,10447]]
    fig, ax = plt.subplots(2, 2)
    demo_plot_P(results_IOB2, ax[0][0], color='r')
    demo_plot_P(results_IOBES, ax[0][0], color='b')
    demo_plot_R(results_IOB2, ax[0][1], color='r')
    demo_plot_R(results_IOBES, ax[0][1], color='b')
    demo_plot_F1(results_IOB2, ax[1][0], color='r', labels=["IOB2 (this study)", "IOB2 (Wang et. al., 2015)"])
    demo_plot_F1(results_IOBES, ax[1][0], color='b', labels=["IOBES (this study)", "IOBES (Wang et. al., 2015)"])
    ax[0][0].set_title("P posterior")
    ax[0][1].set_title("R posterior")
    ax[1][0].set_title("F1 posterior")
    ax[0][0].tick_params(labelsize=12)
    ax[0][1].tick_params(labelsize=12)
    ax[1][0].tick_params(labelsize=12)
    ax[1][1].axis('off')
    ax[1][0].legend(loc='center', frameon=False, numpoints=1, bbox_to_anchor=[1.6, 0.5])
    leg = ax[1][0].get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)
    plt.show()
    pass

if __name__ == "__main__":
    word_seg_task_beta_dist_plots()
    ner_task_beta_dist_plots()
    org_ner_task_beta_dist_plots()
    pass