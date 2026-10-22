import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. Load data
# ==========================================
df = pd.read_csv("Titanic.csv")

print(df.head())
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# ==========================================
# 2. Handle missing values
# ==========================================
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])  # 77% missing, too sparse to fill

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# ==========================================
# 3. Encode categorical columns
# ==========================================
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop identifier columns with no predictive value
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# ==========================================
# 4. Split features and target
# ==========================================
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X = df[feature_cols]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 5. Scale features (fit on train only, apply to both)
# ==========================================
scaler = MinMaxScaler(feature_range=(-1, 1))

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

# ==========================================
# 6. Train model
# ==========================================
model = LogisticRegression()
model.fit(X_train, y_train)

# ==========================================
# 7. Evaluate
# ==========================================
y_pred_test = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# ==========================================
# 8. Predict a new passenger (enter your own values)
# ==========================================
print("\nEnter passenger details to predict survival:")

pclass = int(input("Pclass (1, 2, or 3): "))
sex = int(input("Sex (0 = male, 1 = female): "))
age = float(input("Age: "))
sibsp = int(input("SibSp (# siblings/spouses aboard): "))
parch = int(input("Parch (# parents/children aboard): "))
fare = float(input("Fare: "))
embarked = int(input("Embarked (0 = S, 1 = C, 2 = Q): "))

new_passenger = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

new_passenger_scaled = pd.DataFrame(
    scaler.transform(new_passenger),
    columns=feature_cols
)

prediction = model.predict(new_passenger_scaled)
probability = model.predict_proba(new_passenger_scaled)

print("\nPredicted survival (0 = died, 1 = survived):", prediction[0])
print("Probability [died, survived]:", probability[0])