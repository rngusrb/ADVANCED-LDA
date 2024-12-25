import ssl
import os
import numpy as np
import numpy.linalg as LA
import time
import cv2
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

def getImage(category, kind=None, flatten=True):
    image_list = []
    category_path = os.path.join(category, kind) if kind else category

    if not os.path.exists(category_path):
        print(f"Path does not exist: {category_path}")
        return np.array(image_list)

    for root, dirs, files in os.walk(category_path):
        for filename in files:
            path = os.path.join(root, filename)

            image = cv2.imread(path)
            if isinstance(image, np.ndarray):
                image = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                if flatten:
                    image = image.flatten()

                image_list.append(image)

    return np.array(image_list)

def display_images(images, num_images=10):
    plt.figure(figsize=(20, 10))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(64, 64), cmap='gray')
        plt.axis('off')

    plt.show()

# 학습 데이터로 LDA 모델 학습
print("Fitting LDA model with total train images...")
train_category = 'images/train'
kinds = ['angry', 'disgust', 'fear', 'happy']

images_dict = {}
labels_dict = {}
label_mapping = {kind: idx for idx, kind in enumerate(kinds)}

print("Loading and processing training images...")
for kind in kinds:
    print(f"Loading {kind} images...")
    images_dict[kind] = getImage(train_category, kind)
    labels_dict[kind] = np.full(images_dict[kind].shape[0], label_mapping[kind])

# 학습 데이터 합치기
X = np.vstack([images_dict[kind] for kind in kinds])
y = np.concatenate([labels_dict[kind] for kind in kinds])

validation_category = 'images/validation'
validation_images_dict = {}
validation_labels_dict = {}

print("Loading and processing validation images...")
for kind in kinds:
    print(f"Loading {kind} validation images...")
    validation_images_dict[kind] = getImage(validation_category, kind)
    validation_labels_dict[kind] = np.full(validation_images_dict[kind].shape[0], label_mapping[kind])

# 검증 데이터 합치기
X_validation = np.vstack([validation_images_dict[kind] for kind in kinds])
y_validation = np.concatenate([validation_labels_dict[kind] for kind in kinds])
##########################################

def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    epsilon=1/1000
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        cal1_vet=[]
        sum_vet=[]
        beta=np.ones((X[y==cl].shape[1]))*epsilon
        for data1 in X[y==cl]:
            cal_vet=1/(np.abs(data1-np.median(X[y==cl], axis=0))+beta)
            cal1_vet.append(cal_vet*(data1))
            sum_vet.append(cal_vet)
        mean_vectors.append(np.sum(cal1_vet,axis=0)/np.sum(sum_vet,axis=0))
    return mean_vectors

def scatter_within(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += ((row - mv) / LA.norm(row - mv)).dot(((row - mv) / LA.norm(row - mv)).T)
        S_W += class_sc_mat
    return S_W

def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * ((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).dot(((mean_vec - overall_mean) / LA.norm(mean_vec - overall_mean)).T)
    return S_B

def get_components(eig_vals, eig_vecs, n_comp=2):
    # 고유값을 내림차순으로 정렬
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # 가장 큰 고유값에 해당하는 고유벡터 선택 (k개의 축 선택)
    W = np.hstack([eig_pairs[i][1].reshape(X.shape[1], 1) for i in range(len(np.unique(y)) - 1)])
    return W

S_W, S_B = scatter_within(X, y), scatter_between(X, y)
trace_w = np.trace(S_W)
trace_b = np.trace(S_B)
eig_vals, eig_vecs =0,0
if trace_w>10*trace_b:
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W))
elif trace_w<0.1*trace_b:
    eig_vals, eig_vecs = np.linalg.eig(S_B)
else:
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

W = get_components(eig_vals, eig_vecs, n_comp=2)
print('EigVals: %s\n\nEigVecs: %s' % (eig_vals, eig_vecs))
print('\nW: %s' % W)

X_lda = X.dot(W)


def predict_lda(X, W, X_train, y_train, k=5):
    if len(X) == 0:
        print("No images found for prediction.")
        return []

    print("Predicting using LDA model...")
    start_time = time.time()

    X_lda = X.dot(W)
    X_train_lda = X_train.dot(W)

    # kNN 알고리즘을 직접 구현하여 예측
    def knn_predict(X_train, y_train, X_test, k=5):
        from collections import Counter
        predictions = []
        for test_point in X_test:
            distances = np.linalg.norm(X_train - test_point, axis=1)
            nearest_neighbors = np.argsort(distances)[:k]
            nearest_labels = y_train[nearest_neighbors]
            common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(common_label)
        return np.array(predictions)

    predictions = knn_predict(X_train_lda, y_train, X_lda, k)

    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds.")
    return predictions

##########################################

y_pred = predict_lda(X_validation, W, X, y)
accuracy = accuracy_score(y_validation, y_pred)
print(f'LDA Classification Accuracy on validation set: {accuracy * 100:.2f}%')
print(classification_report(y_validation, y_pred, target_names=kinds))
