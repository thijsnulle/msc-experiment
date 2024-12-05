import glob
from collections import Counter

import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
sns.color_palette("colorblind")

def power_law_distribution_unique_solutions():
    unique_solutions_counter = Counter()

    for file in glob.glob('data/power-law-distribution/**/*.txt'):
        with open(file) as f:
            num_unique_solutions = len(set([x for x in f.read().splitlines() if x.strip()]))
            
            if num_unique_solutions > 0:
                unique_solutions_counter[num_unique_solutions] += 1

    fig, ax = plt.subplots(figsize=(8,4))

    sns.histplot(ax=ax, data=list(unique_solutions_counter.elements()), bins=100, stat='proportion')

    plt.xlim(xmin=1)
    plt.title('Distribution of Total Unique Generations per Line')
    plt.xlabel('Number of Unique Generations')
    plt.tight_layout()
    plt.savefig('output/distribution-unique-line-generations')

    fig, ax = plt.subplots(figsize=(12,4))

    sns.histplot(ax=ax, data=list(unique_solutions_counter.elements()), bins=100, stat='proportion')

    plt.xlim(xmin=1)
    plt.title('Distribution of Total Unique Generations per Line')
    plt.xlabel('Number of Unique Generations')
    plt.tight_layout()
    plt.savefig('output/distribution-unique-line-generations-wide')

if __name__ == '__main__':
    power_law_distribution_unique_solutions()

