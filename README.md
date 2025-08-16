<!-- @format -->

# Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using historical passenger data. This project demonstrates data preprocessing, feature engineering, and machine learning classification techniques.

## 🎯 Project Overview

This project analyzes the famous Titanic dataset to build a predictive model that determines whether a passenger would have survived the Titanic disaster based on various features such as age, gender, passenger class, fare, and family size.

## 🚀 Features

- **Data Preprocessing**: Handles missing values and feature engineering
- **Feature Selection**: Removes irrelevant columns and focuses on predictive features
- **Machine Learning Model**: Uses K-Nearest Neighbors (KNN) classifier
- **Model Optimization**: Implements GridSearchCV for hyperparameter tuning
- **Performance Evaluation**: Provides accuracy metrics and confusion matrix
- **Data Visualization**: Creates insightful plots using matplotlib and seaborn

## 📦 Packages Used

### Core Data Science

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing and array operations

### Machine Learning

- **scikit-learn** - Machine learning algorithms and tools
  - `train_test_split` - Dataset splitting for training/testing
  - `GridSearchCV` - Hyperparameter optimization
  - `MinMaxScaler` - Feature scaling
  - `KNeighborsClassifier` - KNN classification algorithm
  - `accuracy_score` - Model accuracy evaluation
  - `confusion_matrix` - Classification performance metrics

### Visualization

- **matplotlib** - Basic plotting and visualization
- **seaborn** - Statistical data visualization

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd machinelearningproj
   ```

2. **Create a virtual environment**

   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project uses the Titanic dataset containing passenger information:

- **PassengerId**: Unique identifier
- **Survived**: Target variable (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation

## 🔧 Usage

1. **Run the main script**

   ```bash
   python main.py
   ```

2. **The script will:**
   - Load and explore the Titanic dataset
   - Preprocess the data (handle missing values, feature engineering)
   - Train the machine learning model
   - Evaluate model performance
   - Display results and visualizations

## 📈 Model Performance

The KNN classifier is optimized using GridSearchCV to find the best hyperparameters. The model's performance is evaluated using:

- Accuracy score
- Confusion matrix
- Cross-validation results

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📚 Resources

- [Titanic Dataset Documentation](https://www.kaggle.com/c/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**Note**: This project is for educational purposes and demonstrates fundamental machine learning concepts using a well-known dataset.
