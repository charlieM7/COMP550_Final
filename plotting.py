import random

import matplotlib.pyplot as plt

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
        if polarity[i] >= 0.5:
            positive_pol.append(polarity[i])
            positive_eng.append(engagement[i])
            # positive[polarity[i]] = engagement[i]
        elif polarity[i] <= -0.5:
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

        if polarity[i] >= 0.5:
            # is positive
            pos_sum += engagement[i]
            pos_count += 1
        elif polarity[i] <= -0.5:
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
    plt.plot(polarity, engagement, 'ro')
    plt.title("Polarity vs Engagement")
    plt.xlabel("polarity")
    plt.ylabel("engagement")
    plt.show()
    # plt.savefig("scatter_plot.jpg")

def scatter_plot_abs(polarity, engagement):
    # convert neg value to positive
    # now only check if polarity influence engagement

    neu_x = get_abs(neutral_pol)

    neg_x = get_abs(negative_pol)

    plt.scatter(positive_pol, positive_eng, color="red", label="positive")
    plt.scatter(neg_x, negative_eng, color="blue", label="negative")
    plt.scatter(neu_x, neutral_eng, color="black", label="neutral")

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot 1')

    plt.show()



def bar_plot_means(polarity, engagement):
    means = get_mean(polarity, engagement)
    # ['positive', 'neutral', 'negative']
    names = means.keys()
    # respective average of 3 categories
    values = means.values()
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(9, 3))

    # plt.subplot(131)
    plt.bar(names, values, color=colors)
    plt.title('Mean of positive, neutral, and negative tagged text')
    plt.show()
    # plt.savefig("means.png")


if __name__ == "__main__":
    polarity = gen_rand_list(50, 2)
    engagement = gen_rand_list(50, 10)

    # polarity = [0.5, 1, 0, 0.5, -1, -0.25]
    # engagement = [1, -5, 1.5, 2, 4, 5]
    sort_pos_neu_neg(polarity, engagement)

    print(len(positive_pol))
    print(len(positive_eng))
    print()
    print(len(negative_pol))
    print(len(negative_eng))
    scatter_plot(polarity, engagement)
    bar_plot_means(polarity, engagement)
    scatter_plot_abs(polarity, engagement)
