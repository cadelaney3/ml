import numpy as np
import matplotlib.pyplot as plt

def get_num_words_per_sample(sample_texts):
    """Returns the median number of words per sample given corpus.

    # Arguments
        sample_texts: list, sample texts.

    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

def get_num_samples_per_class(sample_labels):
    count_pos = 0
    count_neg = 0
    for label in sample_labels:
        if label == 1:
            count_pos += 1
        else:
            count_neg += 1

    return (count_pos, count_neg)

def get_num_classes(sample_labels):
    num_classes = []
    for label in np.nditer(sample_labels):
        if label not in num_classes:
            num_classes.append(label)
    
    return len(num_classes)


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.savefig("sample_length_distrib.png")

def plot_word_frequency_dist(sample_texts):
    word_count_dict = {}
    for s in sample_texts:
        #for word in s:
        s = s.split()
        for word in s:
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
    print(word_count_dict)
    '''
    objects = word_count_dict.keys()
    y_pos = np.arange(len(objects))
    counts = word_count_dict.values()
    plt.bar(y_pos, counts, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('frequencies')
    plt.xlabel('N-grams')
    plt.title('Frequency distribution of n-grams')
    plt.savefig('frequency_distrib.png')
    '''

'''
median_num_words_per_sample = get_num_words_per_sample(train_texts)
samples_per_class = get_num_samples_per_class(train_labels)
print(median_num_words_per_sample)
print(samples_per_class)
plot_sample_length_distribution(train_texts)
'''
# plot_word_frequency_dist(train_texts)