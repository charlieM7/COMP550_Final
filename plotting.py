import matplotlib.pyplot as plt


def plot(polarity, engagement):
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'color')
    plt.plot(polarity, engagement, 'ro')
    plt.show()

def bar_plot():
    pos_sum = 0
    neu_sum = 0
    neg_sum = 0
    # for i in range (len(polarity)):
    #   if polarity[i] >= 0.5:
    #   pos_sum += engagement[i]
    #repeat

    names = ['positive', 'neutral', 'negative']
    # respective average of 3 categories
    values = [1, 10, 100]

    plt.figure(figsize=(9, 3))

    plt.subplot(131)
    plt.bar(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()