import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from scipy.stats import spearmanr, pearsonr
import pandas as pd


df = pd.read_csv('data/analysis_data1.csv', sep=',', header=0)
# print(df.values)


# polarity = [0.5, 1, 0, 0.5, -1, -0.25]
# engagement = [1, -5, 1, 2, 4, 5]

# { polarity_score : engagement_score}
positive_pol = []
positive_eng = []

neutral_pol = []
neutral_eng = []

negative_pol = []
negative_eng = []



def gen_rand_list(number, val):
    my_list = []
    for i in range(number):
        my_list.append(val * random.random() - val/2)
    return my_list


def sort_pos_neu_neg(polarity, engagement):
    for i in range(len(polarity)):
        if polarity[i] >= 0.05:
            positive_pol.append(polarity[i])
            positive_eng.append(engagement[i])
            # positive[polarity[i]] = engagement[i]
        elif polarity[i] <= -0.05:
            negative_pol.append(polarity[i])
            negative_eng.append(engagement[i])
            # negative[polarity[i]] = engagement[i]
        else:
            neutral_pol.append(polarity[i])
            neutral_eng.append(engagement[i])
            # neutral[polarity[i]] = engagement[i]


def get_abs (list):
    ans = []
    for key in list:
        ans.append(abs(key))
    return ans


def get_mean(polarity, engagement):
    pos_sum = 0
    pos_count = 0
    neu_sum = 0
    neu_count = 0
    neg_sum = 0
    neg_count = 0

    for i in range(len(polarity)):

        if polarity[i] >= 0.05:
            # is positive
            pos_sum += engagement[i]
            pos_count += 1
        elif polarity[i] <= -0.05:
            # is negative
            neg_sum += engagement[i]
            neg_count += 1
        else:
            neu_sum += engagement[i]
            neu_count += 1

    pos_mean = pos_sum / pos_count
    neu_mean = neu_sum / neu_count
    neg_mean = neg_sum / neg_count
    means = {"positive": pos_mean, "neutral": neu_mean, "negative": neg_mean}
    return means


def scatter_plot(polarity, engagement):
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'color')
    plt.scatter(polarity, engagement)
    plt.title("Polarity vs Engagement")
    plt.xlabel("polarity")
    plt.ylabel("engagement")

    plt.draw()
    plt.savefig("scatter_plot.jpg")
    plt.show()

def scatter_plot_abs(polarity, engagement):
    # convert neg value to positive
    # now only check if polarity influence engagement

    neu_x = get_abs(neutral_pol)

    neg_x = get_abs(negative_pol)

    plt.scatter(positive_pol, positive_eng, color="red", label="positive")
    plt.scatter(neg_x, negative_eng, color="blue", label="negative")
    plt.scatter(neu_x, neutral_eng, color="black", label="neutral")

    plt.xlabel('polarity (absolute value)')
    plt.ylabel('engagement')
    plt.title('Scatter Plot 1')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
    plt.draw()
    plt.savefig("scatter_plot_abs.jpg")
    plt.show()



def get_co_var(polarity, engagement):
    covar_matrix = np.cov(polarity, engagement)
    print("Polarity variance: " + covar_matrix[0][0].astype('|S6').decode('UTF-8'))
    print("Engagement variance: " + covar_matrix[1][1].astype('|S6').decode('UTF-8'))
    print("Polarity-Engagement covariance: " + covar_matrix[0][1].astype('|S6').decode('UTF-8'))
    corr = pearsonr(polarity, engagement)
    print("Pearson Correlation: ")
    print(corr)
    return covar_matrix


def bar_plot_means(polarity, engagement):
    means = get_mean(polarity, engagement)
    # ['positive', 'neutral', 'negative']
    names = means.keys()
    # respective average of 3 categories
    values = means.values()
    colors = ['red', 'black', 'blue']

    plt.figure(figsize=(9, 3))

    # plt.subplot(131)
    plt.bar(names, values, color=colors)
    plt.title('Mean of positive, neutral, and negative tagged text')

    plt.draw()
    plt.savefig("means.png")
    plt.show()



if __name__ == "__main__":
    # polarity = gen_rand_list(50, 2)
    # engagement = gen_rand_list(50, 10)
    # print( list(df.columns.values))

    polarity = df['Sentiment'].to_numpy()
    engagement = df['Score'].to_numpy()
    comments = df['Num_Comments'].to_numpy()
    # print(polarity)
    # print(engagement)
    # polarity = [0.5, 1, 0, 0.5, -1, -0.25]
    # engagement = [1, -5, 1.5, 2, 4, 5]
    sort_pos_neu_neg(polarity, engagement)

    print(len(positive_pol))
    print(len(positive_eng))
    print()
    print(len(negative_pol))
    print(len(negative_eng))
    print()
    print("Overall")
    print(get_co_var(polarity,engagement))
    print()
    print("Negative")
    print(get_co_var(negative_pol, negative_eng))
    print()
    print("Positive")
    print(get_co_var(positive_pol, positive_eng))
    print()
    print("Neutral")
    print(get_co_var(neutral_pol, neutral_eng))
    print()
    scatter_plot(polarity, engagement)
    bar_plot_means(polarity, engagement)
    scatter_plot_abs(polarity, engagement)
    print(spearmanr(polarity, engagement))

    print(get_co_var(get_abs(polarity),engagement))
    print(spearmanr(get_abs(polarity),engagement))
    #scatter_plot_abs(polarity, comments)
    # bar_plot_means(polarity, comments)
