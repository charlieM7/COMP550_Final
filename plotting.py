import matplotlib.pyplot as plt


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
    polarity = [0.5, 1, 0, 0.5, -1, -0.25]
    engagement = [1, -5, 1, 2, 4, 5]
    scatter_plot(polarity, engagement)
    bar_plot_means(polarity, engagement)
