import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = 48.9115179061889, 0.896383164343276
    device0_mean, device0_std =  27.781615877151488, 0.27602309691571497
    device1_mean, device1_std =  27.569580149650573, 0.29366889905848914
    plot([device0_mean, device1_mean, single_mean],
        [device0_std, device1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'training_time_ddp_vs_rn.png')
    

    dp_mean, dp_std = 81479.17091235226+81978.17690015228, 515.1720176845024  # Data parallel - 2 GPUs
    single_mean, single_std = 83286.7432855713, 214.67832216425958  # Single GPU
    means = [dp_mean, single_mean]
    stds = [dp_std, single_std]
    labels = ['Data Parallel - 2GPUs', 'Single GPU']
    plot(means, stds, labels, 'tokens_per_second_comparison_ddp_vs_rn.png')


    pp_mean, pp_std = 40.137168765068054, 0.607704758644104
    mp_mean, mp_std = 52.65492844581604, 3.762629508972168
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'training_time_pp_vs_mp.png')
    
    pp_mean, pp_std = 15948.976158787815, 241.47863452770616
    mp_mean, mp_std = 12216.991135376493, 873.0049154681092
    plot([pp_mean, mp_mean],
        [pp_std, mp_std],
        ['Pipeline Parallel', 'Model Parallel'],
        'tokens_per_second_comparison_pp_vs_mp.png')