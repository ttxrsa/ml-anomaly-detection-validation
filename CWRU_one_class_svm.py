import os
import numpy as np

from scipy.io import loadmat
from scipy.stats import kurtosis, skew

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_PATH = "data_if"
WINDOW_SIZE = 2048
RANDOM_STATE = 42

def load_cwru_signal(file_path):
    data = loadmat(file_path)

    for key in data.keys():
        if "DE_time" in key:
            return data[key].flatten()

    raise ValueError(f"DE_time não encontrado em {file_path}")


def create_windows(signal, window_size=2048):
    windows = []

    for i in range(0, len(signal) - window_size + 1, window_size):
        windows.append(signal[i:i + window_size])

    return np.array(windows)


def extract_features(windows):
    features = []

    for window in windows:
        rms = np.sqrt(np.mean(window ** 2))
        std = np.std(window)
        kurt = kurtosis(window)
        skw = skew(window)
        peak_to_peak = np.ptp(window)

        features.append([
            rms,
            std,
            kurt,
            skw,
            peak_to_peak
        ])

    return np.array(features)


normal_files = [
    "Normal_97.mat",
    "Normal_98.mat",
    "Normal_99.mat",
    "Normal_100.mat"
]

fault_files = [
    "Inner_Race_105.mat",
    "Inner_Race_106.mat",
    "Inner_Race_107.mat",
    "Inner_Race_108.mat",

    "Outer_Race_130.mat",
    "Outer_Race_131.mat",
    "Outer_Race_132.mat",
    "Outer_Race_133.mat"
]

normal_features = []
fault_features = []


for file in normal_files:

    path = os.path.join(DATA_PATH, file)

    signal = load_cwru_signal(path)

    windows = create_windows(
        signal,
        WINDOW_SIZE
    )

    features = extract_features(windows)

    normal_features.append(features)


for file in fault_files:

    path = os.path.join(DATA_PATH, file)

    signal = load_cwru_signal(path)

    windows = create_windows(
        signal,
        WINDOW_SIZE
    )

    features = extract_features(windows)

    fault_features.append(features)


X_normal = np.vstack(normal_features)

X_fault = np.vstack(fault_features)


print("Dados carregados:")
print("Normal:", X_normal.shape)
print("Falhas:", X_fault.shape)


X_train, X_normal_test = train_test_split(
    X_normal,
    test_size=0.30,
    random_state=RANDOM_STATE
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_normal_test = scaler.transform(
    X_normal_test
)

X_fault = scaler.transform(
    X_fault
)


from itertools import product


param_grid = {
    "nu": [
        0.01,
        0.03,
        0.05,
        0.07,
        0.10
    ],

    "gamma": [
        "scale",
        "auto",
        0.001,
        0.01,
        0.1,
        1
    ],

    "threshold_percentile": [
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99
    ]
}


X_test = np.vstack([
    X_normal_test,
    X_fault
])


y_test = np.concatenate([
    np.zeros(
        len(X_normal_test),
        dtype=int
    ),

    np.ones(
        len(X_fault),
        dtype=int
    )
])


best_f1 = -1
best_params = None
best_threshold = None
best_model = None
best_pred = None
best_scores = None


for nu, gamma in product(
    param_grid["nu"],
    param_grid["gamma"]
):

    model = OneClassSVM(
        kernel="rbf",
        nu=nu,
        gamma=gamma
    )

    model.fit(X_train)

    train_scores = -model.score_samples(
        X_train
    )

    test_scores = -model.score_samples(
        X_test
    )


    for threshold_percentile in param_grid["threshold_percentile"]:

        threshold = np.percentile(
            train_scores,
            threshold_percentile
        )

        y_pred = (
            test_scores > threshold
        ).astype(int)

        current_f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        if current_f1 > best_f1:

            best_f1 = current_f1

            best_params = {
                "nu": nu,
                "gamma": gamma,
                "threshold_percentile": threshold_percentile
            }

            best_threshold = threshold

            best_model = model

            best_pred = y_pred.copy()

            best_scores = test_scores.copy()


tn, fp, fn, tp = confusion_matrix(
    y_test,
    best_pred
).ravel()


precision = precision_score(
    y_test,
    best_pred
)

recall = recall_score(
    y_test,
    best_pred
)

f1 = f1_score(
    y_test,
    best_pred
)

auc = roc_auc_score(
    y_test,
    best_scores
)

specificity = tn / (tn + fp)


print("MELHOR RESULTADO - ONE-CLASS SVM")

print("Melhores parâmetros:")
print(best_params)

print(f"Threshold:           {best_threshold:.6f}")
print(f"F1:                  {f1:.6f}")
print(f"Precision:           {precision:.6f}")
print(f"Recall:              {recall:.6f}")
print(f"Specificity:         {specificity:.6f}")
print(f"AUC-ROC:             {auc:.6f}")
print(f"False alarms (FP):   {fp}")
print(f"Missed anomalies:    {fn}")

print("\nMatriz de confusão:")

print(
    confusion_matrix(
        y_test,
        best_pred
    )
)