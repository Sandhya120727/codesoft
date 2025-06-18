import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load with latin1 encoding to avoid errors
df = pd.read_csv("movie rating prediction.csv", encoding='latin1')

print(df.columns)

# Define feature columns and target using actual CSV column names
feature_cols = ['Actor 1', 'Actor 2', 'Actor 3', 'Director']
target_col = 'Rating'

# Fill missing categorical columns with 'Unknown'
for col in feature_cols:
    df[col].fillna('Unknown', inplace=True)

# For target (Rating) fill missing with mean (or drop rows with missing rating)
df[target_col].fillna(df[target_col].mean(), inplace=True)

X = df[feature_cols]
y = df[target_col]

# Preprocess categorical features with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R^2 score on test set: {score:.3f}")

def predict_rating():
    print("\nEnter movie details to predict IMDb rating:")
    actor_1 = input("Actor 1: ").strip()
    actor_2 = input("Actor 2: ").strip()
    actor_3 = input("Actor 3: ").strip()
    director = input("Director: ").strip()

    input_df = pd.DataFrame({
        'Actor 1': [actor_1],
        'Actor 2': [actor_2],
        'Actor 3': [actor_3],
        'Director': [director]
    })

    pred = model.predict(input_df)[0]
    print(f"\nPredicted IMDb rating: {pred:.2f}")

predict_rating()

