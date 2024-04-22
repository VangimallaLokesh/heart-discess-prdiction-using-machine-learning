import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.utils import Bunch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the first dataset
url1 = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"
data1 = pd.read_csv(url1)

# Drop rows with missing target values
data1_with_names = data1.dropna(subset=['target'])

# Perform one-hot encoding for categorical variables in dataset 1


# Split dataset 1 into features and target variable
X1 = data1.drop('target', axis=1)
y1 = data1['target']

# Create a dataset object with feature names for dataset 1
data1_with_names = Bunch(data=X1, target=y1, feature_names=X1.columns.tolist(), target_names=['target'])

# Split data 1 for training and testing
X1_train, X1_test, y1_train, y1_test = train_test_split(data1_with_names.data, data1_with_names.target, test_size=0.2, random_state=42)

# Train classifiers for dataset 1
rf_classifier1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier1.fit(X1_train, y1_train)

svm_classifier1 = SVC(kernel='rbf', gamma='auto', random_state=42)
svm_classifier1.fit(X1_train, y1_train)

dt_classifier1 = DecisionTreeClassifier(random_state=42)
dt_classifier1.fit(X1_train, y1_train)

knn_classifier1 = KNeighborsClassifier()
knn_classifier1.fit(X1_train, y1_train)

nb_classifier1 = GaussianNB()
nb_classifier1.fit(X1_train, y1_train)

gb_classifier1 = GradientBoostingClassifier(random_state=42)
gb_classifier1.fit(X1_train, y1_train)

# Load the second dataset
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data2 = pd.read_csv(url2, names=column_names, na_values='?')

# Drop rows with missing target values
data2_with_names = data2.dropna(subset=['target'])

# Perform one-hot encoding for categorical variables in dataset 2

# Split dataset 2 into features and target variable
X2 = data2.drop('target', axis=1)
y2 = data2['target']

# Fill missing values in dataset 2
X2.fillna(X2.mean(), inplace=True)

# Create a dataset object with feature names for dataset 2
data2_with_names = Bunch(data=X2, target=y2, feature_names=X2.columns.tolist(), target_names=['HeartDisease'])

# Split data 2 for training and testing
X2_train, X2_test, y2_train, y2_test = train_test_split(data2_with_names.data, data2_with_names.target, test_size=0.2, random_state=42)

# Train classifiers for dataset 2
rf_classifier2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier2.fit(X2_train, y2_train)

svm_classifier2 = SVC(kernel='rbf', gamma='auto', random_state=42)
svm_classifier2.fit(X2_train, y2_train)

dt_classifier2 = DecisionTreeClassifier(random_state=42)
dt_classifier2.fit(X2_train, y2_train)

knn_classifier2 = KNeighborsClassifier()
knn_classifier2.fit(X2_train, y2_train)

nb_classifier2 = GaussianNB()
nb_classifier2.fit(X2_train, y2_train)

gb_classifier2 = GradientBoostingClassifier(random_state=42)
gb_classifier2.fit(X2_train, y2_train)

# Attribute names for dataset 1
attribute_names1 = X1.columns.tolist()

# Attribute names for dataset 2
attribute_names2 = X2.columns.tolist()

# Calculate evaluation metrics for classifiers for dataset 1
def calculate_metrics(classifier, X_test, y_test):
    accuracy = accuracy_score(y_test, classifier.predict(X_test))
    precision = precision_score(y_test, classifier.predict(X_test), average='macro')
    recall = recall_score(y_test, classifier.predict(X_test), average='macro')
    f1 = f1_score(y_test, classifier.predict(X_test), average='macro')
    f2 = fbeta_score(y_test, classifier.predict(X_test), average='macro', beta=2)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2}

metrics_rf1 = calculate_metrics(rf_classifier1, X1_test, y1_test)
metrics_svm1 = calculate_metrics(svm_classifier1, X1_test, y1_test)
metrics_dt1 = calculate_metrics(dt_classifier1, X1_test, y1_test)
metrics_knn1 = calculate_metrics(knn_classifier1, X1_test, y1_test)
metrics_nb1 = calculate_metrics(nb_classifier1, X1_test, y1_test)
metrics_gb1 = calculate_metrics(gb_classifier1, X1_test, y1_test)

# Calculate evaluation metrics for classifiers for dataset 2
metrics_rf2 = calculate_metrics(rf_classifier2, X2_test, y2_test)
metrics_svm2 = calculate_metrics(svm_classifier2, X2_test, y2_test)
metrics_dt2 = calculate_metrics(dt_classifier2, X2_test, y2_test)
metrics_knn2 = calculate_metrics(knn_classifier2, X2_test, y2_test)
metrics_nb2 = calculate_metrics(nb_classifier2, X2_test, y2_test)
metrics_gb2 = calculate_metrics(gb_classifier2, X2_test, y2_test)

# Display bar graph for evaluation scores of classifiers for dataset 1
def show_bar_graph1(metrics1, metrics2, metrics3, metrics4, metrics5, metrics6):
    labels = list(metrics1.keys())
    scores1 = list(metrics1.values())
    scores2 = list(metrics2.values())
    scores3 = list(metrics3.values())
    scores4 = list(metrics4.values())
    scores5 = list(metrics5.values())
    scores6 = list(metrics6.values())

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width, scores1, width, label='Random Forest')
    rects2 = ax.bar(x - 2*width, scores2, width, label='SVM')
    rects3 = ax.bar    (x - width, scores3, width, label='Decision Tree')
    rects4 = ax.bar(x, scores4, width, label='KNN')
    rects5 = ax.bar(x + width, scores5, width, label='Naive Bayes')
    rects6 = ax.bar(x + 2*width, scores6, width, label='Gradient Boosting')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Scores for Classifiers - Dataset 1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Display bar graph for evaluation scores of classifiers for dataset 2
def show_bar_graph2(metrics1, metrics2, metrics3, metrics4, metrics5, metrics6):
    labels = list(metrics1.keys())
    scores1 = list(metrics1.values())
    scores2 = list(metrics2.values())
    scores3 = list(metrics3.values())
    scores4 = list(metrics4.values())
    scores5 = list(metrics5.values())
    scores6 = list(metrics6.values())

    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width, scores1, width, label='Random Forest')
    rects2 = ax.bar(x - 2*width, scores2, width, label='SVM')
    rects3 = ax.bar(x - width, scores3, width, label='Decision Tree')
    rects4 = ax.bar(x, scores4, width, label='KNN')
    rects5 = ax.bar(x + width, scores5, width, label='Naive Bayes')
    rects6 = ax.bar(x + 2*width, scores6, width, label='Gradient Boosting')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Scores for Classifiers - Dataset 2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Display bar graphs for evaluation scores of classifiers for dataset 1
show_bar_graph1(metrics_rf1, metrics_svm1, metrics_dt1, metrics_knn1, metrics_nb1, metrics_gb1)

# Display bar graphs for evaluation scores of classifiers for dataset 2
show_bar_graph2(metrics_rf2, metrics_svm2, metrics_dt2, metrics_knn2, metrics_nb2, metrics_gb2)

# GUI for dataset 1
def predict_and_save1(entries1):
    user_input1 = [float(entry1.get()) for entry1 in entries1]
    user_input1 = np.array(user_input1).reshape(1, -1)
    
    # Make predictions using classifiers for dataset 1
    prediction_rf1 = rf_classifier1.predict(user_input1)[0]
    prediction_svm1 = svm_classifier1.predict(user_input1)[0]
    prediction_dt1 = dt_classifier1.predict(user_input1)[0]
    prediction_knn1 = knn_classifier1.predict(user_input1)[0]
    prediction_nb1 = nb_classifier1.predict(user_input1)[0]
    prediction_gb1 = gb_classifier1.predict(user_input1)[0]

    messagebox.showinfo("Prediction - Dataset 1", f"Random Forest Prediction: {prediction_rf1}\nSVM Prediction: {prediction_svm1}\nDecision Tree Prediction: {prediction_dt1}\nKNN Prediction: {prediction_knn1}\nNaive Bayes Prediction: {prediction_nb1}\nGradient Boosting Prediction: {prediction_gb1}")

    # Show accuracy, precision, recall, f1, and f2 values for dataset 1
    messagebox.showinfo("Evaluation Metrics - Dataset 1", f"Random Forest\nAccuracy: {metrics_rf1['accuracy']}\nPrecision: {metrics_rf1['precision']}\nRecall: {metrics_rf1['recall']}\nF1 Score: {metrics_rf1['f1']}\nF2 Score: {metrics_rf1['f2']}\n\nSVM\nAccuracy: {metrics_svm1['accuracy']}\nPrecision: {metrics_svm1['precision']}\nRecall: {metrics_svm1['recall']}\nF1 Score: {metrics_svm1['f1']}\nF2 Score: {metrics_svm1['f2']}\n\nDecision Tree\nAccuracy: {metrics_dt1['accuracy']}\nPrecision: {metrics_dt1['precision']}\nRecall: {metrics_dt1['recall']}\nF1 Score: {metrics_dt1['f1']}\nF2 Score: {metrics_dt1['f2']}\n\nKNN\nAccuracy: {metrics_knn1['accuracy']}\nPrecision: {metrics_knn1['precision']}\nRecall: {metrics_knn1['recall']}\nF1 Score: {metrics_knn1['f1']}\nF2 Score: {metrics_knn1['f2']}\n\nNaive Bayes\nAccuracy: {metrics_nb1['accuracy']}\nPrecision: {metrics_nb1['precision']}\nRecall: {metrics_nb1['recall']}\nF1 Score: {metrics_nb1['f1']}\nF2 Score: {metrics_nb1['f2']}\n\nGradient Boosting\nAccuracy: {metrics_gb1['accuracy']}\nPrecision: {metrics_gb1['precision']}\nRecall: {metrics_gb1['recall']}\nF1 Score: {metrics_gb1['f1']}\nF2 Score: {metrics_gb1['f2']}")

def makeform1(root1, fields1):
    entries1 = []
    for field1 in fields1:
        row1 = tk.Frame(root1)
        lab1 = tk.Label(row1, width=22, text=field1 + ": ", anchor='w')
        ent1 = tk.Entry(row1)
        ent1.insert(0, "0")
        row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab1.pack(side=tk.LEFT)
        ent1.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries1.append(ent1)
    b1 = tk.Button(root1, text='Predict', command=(lambda e1=entries1: predict_and_save1(e1)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root1, text='Quit', command=root1.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root1.mainloop()

# GUI for dataset 2
def predict_and_save2(entries2):
    user_input2 = [float(entry2.get()) for entry2 in entries2]
    user_input2 = np.array(user_input2).reshape(1, -1)
    
    # Make predictions using classifiers for dataset 2
    prediction_rf2 = rf_classifier2.predict(user_input2)[0]
    prediction_svm2 = svm_classifier2.predict(user_input2)[0]
    prediction_dt2 = dt_classifier2.predict(user_input2)[0]
    prediction_knn2 = knn_classifier2.predict(user_input2)[0]
    prediction_nb2 = nb_classifier2.predict(user_input2)[0]
    prediction_gb2 = gb_classifier2.predict(user_input2)[0]

    messagebox.showinfo("Prediction - Dataset 2", f"Random Forest Prediction: {prediction_rf2}\nSVM Prediction: {prediction_svm2}\nDecision Tree Prediction: {prediction_dt2}\nKNN Prediction: {prediction_knn2}\nNaive Bayes Prediction: {prediction_nb2}\nGradient Boosting Prediction: {prediction_gb2}")

    # Show accuracy, precision, recall, f1, and f2 values for dataset 2
    messagebox.showinfo("Evaluation Metrics - Dataset 2", f"Random Forest\nAccuracy: {metrics_rf2['accuracy']}\nPrecision: {metrics_rf2['precision']}\nRecall: {metrics_rf2['recall']}\nF1 Score: {metrics_rf2['f1']}\nF2 Score: {metrics_rf2['f2']}\n\nSVM\nAccuracy: {metrics_svm2['accuracy']}\nPrecision: {metrics_svm2['precision']}\nRecall: {metrics_svm2['recall']}\nF1 Score: {metrics_svm2['f1']}\nF2 Score: {metrics_svm2['f2']}\n\nDecision Tree\nAccuracy: {metrics_dt2['accuracy']}\nPrecision: {metrics_dt2['precision']}\nRecall: {metrics_dt2['recall']}\nF1 Score: {metrics_dt2['f1']}\nF2 Score: {metrics_dt2['f2']}\n\nKNN\nAccuracy: {metrics_knn2['accuracy']}\nPrecision: {metrics_knn2['precision']}\nRecall: {metrics_knn2['recall']}\nF1 Score: {metrics_knn2['f1']}\nF2 Score: {metrics_knn2['f2']}\n\nNaive Bayes\nAccuracy: {metrics_nb2['accuracy']}\nPrecision: {metrics_nb2['precision']}\nRecall: {metrics_nb2['recall']}\nF1 Score: {metrics_nb2['f1']}\nF2 Score: {metrics_nb2['f2']}\n\nGradient Boosting\nAccuracy: {metrics_gb2['accuracy']}\nPrecision: {metrics_gb2['precision']}\nRecall: {metrics_gb2['recall']}\nF1 Score: {metrics_gb2['f1']}\nF2 Score: {metrics_gb2['f2']}")

def makeform2(root2, fields2):
    entries2 = []
    for field2 in fields2:
        row2 = tk.Frame(root2)
        lab2 = tk.Label(row2, width=22, text=field2 + ": ", anchor='w')
        ent2 = tk.Entry(row2)
        ent2.insert(0, "0")
        row2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab2.pack(side=tk.LEFT)
        ent2.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries2.append(ent2)
    b1 = tk.Button(root2, text='Predict', command=(lambda e2=entries2: predict_and_save2(e2)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root2, text='Quit', command=root2.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root2.mainloop()
    

# Run GUI for dataset 1
root1 = tk.Tk()
root1.title("Heart Disease Prediction - Dataset 1")
makeform1(root1, attribute_names1)

# Run GUI for dataset 2
root2 = tk.Tk()
root2.title("Heart Disease Prediction - Dataset 2")
makeform2(root2, attribute_names2)


