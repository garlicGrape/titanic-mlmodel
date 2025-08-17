# Imports 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('titanic.csv')
data.info()
print(data.isnull().sum()) # will show the number of null values in each column

# Data Cleaning and Feature Engineering
def prepocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Fill missing values for Fare
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

    fill_missing_ages(df)

    # Convert Gender to binary
    df["Sex"] = df["Sex"].map({"male":1, "female":0})

    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0) # 1 if alone, 0 if not
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False) # Creating bins for Fare and Age to catogorize them, creating more columns but wont show in the dataset
    df["AgeBin"] = pd.qcut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)

    return df

# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {} # dictionary for now
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].mean()
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
    axis=1)  # Fill missing ages with class-specific means, keep existing ages if present, axis=1 means apply to each row and look at the Pclass column and Age column

data = prepocess_data(data)

# Create Features / Target Variables (Make Flashcards)
X = data.drop(columns=["Survived"]) # So we cannot see the Survived column in the X variable
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) # X is the front of the flashcard and y is the back of the flashcard 

# ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning
