import matplotlib.pyplot as plt

from target_mc import summarize_stats

if __name__ == "__main__":

    axs = [plt.subplots()[1] for _ in range(18)]

    summarize_stats("results/3cals_20ref_25noref_500iters_215429_10262025", 'hunter_fitter', axes=axs, alpha=0.75)
    summarize_stats("results/3cals_20ref_25noref_500iters_215429_10262025", 'sasha_fitter1', axes=axs, color='r',
                    alpha=0.75)

    plt.show()
