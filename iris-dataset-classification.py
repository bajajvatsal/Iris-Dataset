# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
sns.set()

# Importing the dataset
df = pd.read_csv(r"Iris.csv")
print(df.describe())
print(df.head())
print(df.info())

# Visualization
# Sepal Area Plotting
plt.title("Sepal Parameters Plot")
sns.scatterplot('SepalLengthCm', 'SepalWidthCm', data=df, hue='Species', palette="deep")
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
plt.show()

# Petal Area Plotting
plt.title("Petal Parameters Plot")
sns.scatterplot('PetalLengthCm', 'PetalWidthCm', data=df, hue='Species', palette="deep")
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
plt.show()

# Correlation - Heatmap of the features
df_corr = df.copy()
df_corr = df_corr.drop(["Id", "Species"], axis=1)
corr = df_corr.corr()
sns.heatmap(corr, cmap="Blues",annot=corr)
plt.show()

# Data Preprocessing
# Dropping the Id Column
df.drop(["Id"], axis=1)

# Encoding the categorical data
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Splitting the data into training-test data
X = df.iloc[:, 3:5].values
y = df.iloc[:, -1:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Training the model

# Logistic Regression Classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

# Support Vector Classifier
SVC_classifier = SVC()
SVC_classifier.fit(X_train, y_train)

# Predicting the test-set

# Logistic Regression Classifier
y_pred_lrc = lr_classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred_lrc.reshape(len(y_pred_lrc), 1)), 1))


# Support Vector Classifier
y_pred_svc = SVC_classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred_svc.reshape(len(y_pred_svc), 1)), 1))

# Accuracy Score
accuracy_SVC_model = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy Score of Support Vector Classifier Model: {accuracy_SVC_model}")
accuracy_lrc_model = accuracy_score(y_test, y_pred_lrc)
print(f"Accuracy Score of Logistic Regression Classifier Model: {accuracy_lrc_model}")
