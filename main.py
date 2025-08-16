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

    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

    # Convert Gender to binary
    

