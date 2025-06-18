import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("\nEnter the values for the transaction features to predict if it is fraud or not:")

feature_cols = X.columns.tolist()

user_input = {}

for feature in feature_cols:
    while True:
        try:
            val = float(input(f"Enter value for {feature}: "))
            user_input[feature] = val
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

input_df = pd.DataFrame([user_input])

prediction = model.predict(input_df)[0]

if prediction == 1:
    print("\nPrediction: Fraudulent transaction detected!")
else:
    print("\nPrediction: Transaction is NOT fraudulent.")
