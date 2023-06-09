import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

class WineClassifier:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.preprocess_data()

    def preprocess_data(self):
        self.data['style'] = self.data['style'].replace('red', 0)  # this is required for data training
        self.data['style'] = self.data['style'].replace('white', 1)
        self.y = self.data['style']
        self.X = self.data.drop('style', axis=1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
        self.X_train.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
                                 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
                                 'quality']

    def train_model(self):
        self.model = ExtraTreesClassifier()
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        if self.model is None:
            print("Error: model has not been trained yet!")
            return
        score = self.model.score(self.X_test, self.y_test)
        print("Accuracy:", score)

    def predict(self, wine_characteristics):
        if self.model is None:
            print("Error: model has not been trained yet!")
            return
        return self.model.predict(wine_characteristics).item()

    def get_wine_characteristics(self, database_entry):
        return self.X.iloc[database_entry].values.reshape( 1,-1)

    def get_wine_color(self, database_entry):
        return self.y.iloc[database_entry].item()

    def get_number_of_entries(self):
        return self.data.shape[0]


wine_classifier = WineClassifier("wine_data.csv")
wine_classifier.train_model()
wine_classifier.test_model()


# WineClassifier uses 0 and 1 for wine colors
wine_colors = ["red", "white"]


# Select a random wine from the dataset and predict its color
random_entry = random.choice(range(0, wine_classifier.get_number_of_entries()))
random_wine_characteristics = wine_classifier.get_wine_characteristics(random_entry)
prediction = wine_classifier.predict(random_wine_characteristics)

# Check if prediction is correct
if prediction == wine_classifier.get_wine_color(random_entry):
    print("The prediction was correct! The wine is:", wine_colors[prediction])
else:
    print("The prediction was not correct :(")


# Plotting the bar chart with the highlighted color
counts = wine_classifier.data['style'].value_counts()  # Count the occurrences of each wine color
plt.bar(wine_colors, counts)
plt.xlabel("Wine Color")
plt.ylabel("Frequency")
plt.show()

# Histogram of the 'fixed_acidity' variable
plt.hist(wine_classifier.data['fixed_acidity'], bins=30)
plt.xlabel('fixed_acidity')
plt.ylabel('Frequency')
plt.show()

# Histogram of the 'alcohol' variable
plt.hist(wine_classifier.data['alcohol'], bins=30)
plt.xlabel('alcohol')
plt.ylabel('Frequency')
plt.show()