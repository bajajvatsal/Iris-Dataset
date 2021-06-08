# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Introduction
# %% [markdown]
# Well before starting I wanted you to give a quick outline of the notebook- 
# 
# 
# This is a classic classification problem of iris-flower species.
# In the given dataset we have to do the classification of Iris flower species into three species- Iris-Setosa,Iris-Versicolor and Iris-Virginica
# 
# Parameter used for the classification- SepalLengthCm, SepalWidthCm, PetalLengthCm and PetalWidthCm
# 
# Exploratory Data Analysis include finding distribution of species in among the sepal and petal parameters, correlation between them and so on
# 
# Classification models used are - Logistic Regression, Support Vector Machine, Decision Tree, Naive Bayes
# 
# Parameter used to define the accuracy of the model is accuracy score, since the dataset is too small using K-Fold cross validation is not worth it.
# All the model performs very well on the training dataset, so there's not much too discuss
# 
# Well that's it, give your feedback.
# %% [markdown]
# # Importing the Libraries

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
sns.set()

# %% [markdown]
# # Importing the dataset

# %%
df = pd.read_csv(r"Iris.csv")

# %% [markdown]
# # EDA

# %%
df.head()


# %%
df.describe()


# %%
df.info()


# %%
# Dropping the Id Column
df.drop(["Id"], axis=1)

# %% [markdown]
# ### Sepal Area Plotting

# %%
plt.title("Sepal Parameters Plot")
sns.scatterplot('SepalLengthCm', 'SepalWidthCm', data=df, hue='Species', palette="deep")
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
plt.show()

# %% [markdown]
# ### Petal Area Plotting

# %%
plt.title("Petal Parameters Plot")
sns.scatterplot('PetalLengthCm', 'PetalWidthCm', data=df, hue='Species', palette="deep")
plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1))
plt.show()


# %%
figure, axes = plt.subplots(2, 2,sharex=True,figsize=(11,7))
figure.suptitle('Species distribution among the sepal and petal parameters')
sns.boxplot(ax=axes[0, 0], x="Species",y="PetalLengthCm",data=df)
sns.boxplot(ax=axes[1, 0], x="Species",y="PetalWidthCm",data=df)
sns.boxplot(ax=axes[0, 1], x="Species",y="SepalLengthCm",data=df)
sns.boxplot(ax=axes[1, 1], x="Species",y="SepalWidthCm",data=df)
# axes[0].set_title('first chart with no data')
# axes[1].set_title('second chart with no data')

# %% [markdown]
# # Correlation - Heatmap of the features

# %%
df_copy = df.copy()
df_copy = df_copy.drop(["Id","Species"],axis =1)
corr = df_copy.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(230, 20, as_cmap=True),annot=corr)
plt.show()


# %%
df_copy.hist(figsize=(10,10))
plt.show()

# %% [markdown]
# # Data Preprocessing

# %%
# Encoding the categorical data
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# %% [markdown]
# ## Splitting the data into training-test data
# %% [markdown]
# Here I am only using petal length and width for training the ml models because of their continuos distribution over the different species and the correlation between sepal parameter is low and on the other side the correlation between petal parameter is high , this a fair observation anyone can make after different iterations of model parameter tuning. 

# %%
X = df.iloc[:, 3:5].values
y = df.iloc[:, -1:].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X)

# %% [markdown]
# ## Feature Scaling

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# %% [markdown]
# # Training and Evaluating different ML models
# %% [markdown]
# ## Logistic Regression Classifier

# %%
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train,y_train)
y_pred_lrc = lr_classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred_lrc.reshape(len(y_pred_lrc),1)),1))
accuracy_lrc_model = accuracy_score(y_test, y_pred_lrc)
print(f"Accuracy Score of Logistic Regression Classifier Model: {accuracy_lrc_model}")

# %% [markdown]
# ## Support Vector Classifier

# %%
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_svc = svc_classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred_svc.reshape(len(y_pred_svc),1)),1))
accuracy_SVC_model = accuracy_score(y_test, y_pred_svc)
print(f"Accuracy Score of Support Vector Classifier Model: {accuracy_SVC_model}")

# %% [markdown]
# ## Decision Tree Classifier

# %%
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred_dt.reshape(len(y_pred_dt),1)),1))
dt_acc = accuracy_score(y_test,y_pred_dt)
print(f"Accuracy Score of Descion Tree Classifier Model: {dt_acc}")

# %% [markdown]
# ## Naive Bayes Classifier

# %%
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(X_train,y_train)
y_pred_nb = naive_bayes.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test),1),y_pred_nb.reshape(len(y_pred_nb),1)),1))
naive_bayes_acc = accuracy_score(y_test,y_pred_nb)
print(f"Accuracy Score of Naive Bayes Classifier Model: {dt_acc}")

# %% [markdown]
# # Accuracy Score

# %%
dict_model = {"Logistic Regression":accuracy_lrc_model,"Support Vector":accuracy_SVC_model,"Decision Tree":dt_acc,"Naive Bayes":naive_bayes_acc}


# %%
model_eval = pd.DataFrame()
model_eval.insert(loc = 0,column="Model",value=list(dict_model.keys()))
model_eval.insert(loc=1,column="Accuracy",value=list(dict_model.values()))


# %%
print(model_eval)


# %%

sns.barplot(x=list(model_eval["Model"].values),y=list(model_eval["Accuracy"].values))
plt.xticks(rotation=45)
plt.show()


