import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

csv_file = "iris flower detection.csv"
df = pd.read_csv(csv_file)

print("Columns in CSV:", df.columns.tolist())

species_col_candidates = ['Species', 'species', 'class', 'Class']
species_col = None

for col in species_col_candidates:
    if col in df.columns:
        species_col = col
        break

if species_col is None:
    raise ValueError("Could not find a species/class column in the CSV file.")

print(f"Using species column: '{species_col}'")

le = LabelEncoder()
df[species_col] = le.fit_transform(df[species_col])

drop_cols = []
if 'Id' in df.columns:
    drop_cols.append('Id')
drop_cols.append(species_col)

X = df.drop(drop_cols, axis=1)
y = df[species_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a numeric value.")

print("\nEnter iris flower features to predict species:")

sepal_length = get_float_input("Sepal length (cm): ")
sepal_width = get_float_input("Sepal width (cm): ")
petal_length = get_float_input("Petal length (cm): ")
petal_width = get_float_input("Petal width (cm): ")

user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)

predicted_class_index = clf.predict(user_input)[0]

predicted_species = le.inverse_transform([predicted_class_index])[0]

print(f"\nPredicted iris species: {predicted_species}")
