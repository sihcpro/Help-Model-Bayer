from sklearn.metrics import accuracy_score
import numpy as np


# Read and label for training data
neg_file = open('Neg_Train.txt', 'r', encoding='utf8')
pos_file = open('Pos_Train.txt', 'r', encoding='utf8')
doc_train = []
label_train = []
# doc_train = ['tuyệt_vời', 'nhỏ', 'ít', 'không', 'cũ', 'muỗi', 'tốt', 'nhiệt_tình', 'thân_thiện']
# label_train = [1, 0, 0, 0, 0, 0, 1, 1, 1]
# doc_train = ['nhiệt_tình']
# label_train = [1]

for row in pos_file:
    doc_train.append(row)
    label_train.append(1)

for row in neg_file:
    doc_train.append(row)
    label_train.append(0)


print('Done loading training data')

# Build BOW
set_words = []
for row in doc_train:
    word = row.split(' ')
    set_words += word
set(set_words)
print(len(set_words))

# WordToVec
vectors = []

for row in doc_train:
    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in doc_train:
            vector[i] = 1
    vectors.append(vector)
print(np.shape(vectors))
print('Done building vectors')

# CALCULATE THE PROBABILITY
# Smooth the probability


def smoothing(a, b):
    return float((a+0.1)/(b+0.1))


# General probability
positive = 0
negative = 0
for i in label_train:
    if i == 1:
        positive += 1
    else:
        negative += 1
print(positive, negative)

positive_prob = smoothing(positive, positive + negative)
negative_prob = smoothing(negative, positive + negative)
print(positive_prob, negative_prob)
print('Done General probability')

# Detail probability
bayes_matrix = np.zeros((len(set_words), 4))
for i, word in enumerate(set_words):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for k, v in enumerate(vectors):
        if v[i] == 1:
            if label_train[k] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if label_train[k] == 1:
                FP += 1
            else:
                FN += 1

    bayes_matrix[i][0] = smoothing(TP, positive)
    bayes_matrix[i][1] = smoothing(TN, negative)
    bayes_matrix[i][2] = smoothing(FP, positive)
    bayes_matrix[i][3] = smoothing(FN, negative)
print('Done building matrix')

# TEST


def compare(predict_pos, predict_neg, log):
    while log[0] > log[1]:
        predict_pos /= 10
        log[0] -= 1
        if predict_pos > predict_neg:
            return True

    while log[1] > log[0]:
        predict_neg /= 10
        log[1] -= 1
        if predict_neg > predict_pos:
            return False

    if predict_pos > predict_neg:
        return True
    return False


test_file = open('Review_Test.txt', 'r', encoding='utf8')
# neg_file = open('Neg_Test.txt', 'r', encoding='utf8')
# pos_file = open('Pos_Test.txt', 'r', encoding='utf8')
label_test = []
for i in range(200):
    label_test.append(1)
for i in range(200):
    label_test.append(0)


def predict(review):
    vector = np.zeros(len(set_words))
    for i, word in enumerate(set_words):
        if word in review:
            vector[i] = 1
    log = np.zeros(2)

    predict_pos = positive_prob
    predict_neg = negative_prob

    for i, v in enumerate(vector):
        if v == 0:
            predict_pos *= bayes_matrix[i][2]
            predict_neg *= bayes_matrix[i][3]
        else:
            predict_pos *= bayes_matrix[i][0]
            predict_neg *= bayes_matrix[i][1]

        if predict_pos < 1e-10:
            predict_pos *= 1000
            log[0] += 1

        if predict_neg < 1e-10:
            predict_neg *= 1000
            log[1] += 1

    if compare(predict_pos, predict_neg, log):
        print(review + '===>Pos')
        return 1
        # print('Positive Review')
    else:
        print(review + '===>Neg')
        return 0
        # print('Negative Review')


# Accuracy

pred = [predict(row) for row in test_file]
print(accuracy_score(label_test, pred))

