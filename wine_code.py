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

    def predict(self, data):
        if self.model is None:
            print("Error: model has not been trained yet!")
            return
        return self.model.predict(data)


wine_classifier = WineClassifier("wine_data.csv")
wine_classifier.train_model()
wine_classifier.test_model()


# Load the data from the CSV file
wine_data = pd.read_csv("wine_data.csv")

# Get the number of rows and columns
num_rows = wine_data.shape[0]
num_columns = wine_data.shape[1]

# Print the number of rows and columns
print("Number of rows:", num_rows)
print("Number of columns:", num_columns)

# Here's the code to randomly select a wine from the table and determine if it is red or white:

# Select a random wine
random_wine = random.choice(wine_data['style'])

# Determine if the wine is red or white
if random_wine == 'red':
    print("Here's the code to randomly select a wine from the table : This wine is red.")
else:
    print("Here's the code to randomly select a wine from the table :This wine is white.")



# Predict wine color at index ex.5320
predicted_label = wine_classifier.predict(wine_classifier.X.iloc[5320].values.reshape( 1,-1))

if predicted_label == 0:
    print("The wine at index 5320 is red.")
else:
    print("The wine at index 5320 is white.")


# Count the occurrences of each wine color
wine_colors = ["red", "white"]
counts = wine_classifier.data['style'].value_counts()

# Plotting the bar chart with the highlighted color
plt.bar(wine_colors, counts)
plt.xlabel("Wine Color")
plt.ylabel("Frequency")
plt.show()

# Histogram of the 'fixed_acidity' variable
plt.hist(wine_classifier.data['fixed_acidity'], bins=30)
plt.xlabel('fixed_acidity')
plt.ylabel('Frequency')
plt.show()

# Bar chart of the 'alcohol' variable

plt.bar(wine_colors, counts)
plt.xlabel('Wine Color')
plt.ylabel('alcohol')
plt.show()

