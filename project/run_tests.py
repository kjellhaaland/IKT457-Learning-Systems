from collections import defaultdict
import numpy as np
from keras.datasets import cifar10


def softmax(x):
    if np.sum(np.exp(x), axis=0) == 0:
        return np.exp(x) / 1
    else:
        return np.exp(x) / np.sum(np.exp(x), axis=0)


def minmax(x, i):
    return x[i] / (np.max(x) - np.min(x))


def predict(x, i):
    return minmax(x, i)


def soft_vote(*arrays, weights=None):
    # Convert the arrays to numpy arrays
    arrays = [np.array(arr) for arr in arrays]

    # Check if weights are provided, otherwise set equal weights
    if weights is None:
        weights = [1] * len(arrays)
    else:
        weights = np.array(weights)

    # Weighted average of the predicted probabilities
    soft_voting_result = np.average(arrays, axis=0, weights=weights)

    # Select the class with the highest average probability
    final_prediction = np.argmax(soft_voting_result)

    return final_prediction


def predict_team(X, weights, i):
    # Get the votes for each candidate
    votes = [predict(x, i) for x in X]
    prediction = soft_vote(*votes, weights=weights)
    return prediction


(X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()

Y_train = Y_train.reshape(Y_train.shape[0])
Y_test = Y_test.reshape(Y_test.shape[0])

# Base class sums
Y_ts_threshold = np.loadtxt("class_sums/CIFAR10AdaptiveThresholding_99_2000_500_10.0_10_32_1.txt", delimiter=",")
Y_ts_thermometer_3 = np.loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_3_8_32_1.txt", delimiter=",")
Y_ts_thermometer_4 = np.loadtxt("class_sums/CIFAR10ColorThermometers_99_2000_1500_2.5_4_8_32_1.txt", delimiter=",")
Y_ts_hog = np.loadtxt("class_sums/CIFAR10HistogramOfGradients_99_2000_50_10.0_0_32_0.txt", delimiter=",")

# New class sums
Y_ts_canny = np.loadtxt("class_sums/CIFAR10CannyHoughLines_100_2000_500_10.0_16_32_1.txt", delimiter=",")
Y_ts_otsu = np.loadtxt("class_sums/CIFAR10Otsu_100_2000_500_10.0_16_32_1.txt", delimiter=",")
Y_ts_thresh_inv = np.loadtxt("class_sums/CIFAR10ThresInvBlur_100_2000_500_10.0_16_32_1.txt", delimiter=",")
Y_ts_thermometer_2 = np.loadtxt("class_sums/CIFAR102x2ColorThermometers_100_2000_1500_2.5_2_32_1.txt", delimiter=",")
Y_ts_thermometer_5 = np.loadtxt("class_sums/CIFAR105x5ColorThermometers_100_2000_1500_2.5_5_32_1.txt", delimiter=",")

experts = [
    Y_ts_canny,
    Y_ts_otsu,
    Y_ts_thresh_inv,
    Y_ts_thermometer_2,
    Y_ts_thermometer_5,
    Y_ts_threshold,
    Y_ts_thermometer_3,
    Y_ts_thermometer_4,
    Y_ts_hog,
]

# Weights for each expert
weights = [0.6, 0.6, 1, 1, 1, 2, 2, 2, 4]

# Normalize the weights
weights = [x / sum(weights) for x in weights]

# Variables for accuracy and plotting

predicted_values = np.zeros(Y_test.shape, dtype=np.int32)
correct = 0
total = 0
exps = 0

# To beat: 75.1
# Current best: 76.3

for i in range(Y_ts_threshold.shape[0]):
    total += 1

    vote = predict_team(experts, weights, i)

    predicted_values[i] = vote

    if vote == Y_test[i]:
        correct += 1

print("Team Accuracy: %.1f" % (100 * (correct / total)))
print(f"Correct: {correct}, Total: {total}, Incorrent: {total - correct}")
