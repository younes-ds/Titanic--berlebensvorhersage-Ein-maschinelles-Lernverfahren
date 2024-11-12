import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

# Titanic-Daten laden
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Fehlende Werte auffüllen
df['Age'] = df['Age'].fillna(df['Age'].median())  # Fehlende Alterswerte mit dem Median füllen
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fehlende Embarked-Werte mit dem Modus füllen

# Unnötige Spalten entfernen
df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])

# Neue Features erstellen
df['FamilySize'] = df['SibSp'] + df['Parch']  # SibSp und Parch zu FamilySize kombinieren
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)  # Die alten Spalten löschen

# Kategorische Variablen mit One-Hot-Encoding behandeln
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Numerische Spalten normalisieren (Age, Fare, FamilySize)
scaler = MinMaxScaler()
df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])

# Schiefe der Daten vor der Transformation prüfen
print("\nSchiefe vor der Transformation:")
print(f"Schiefe von Age: {skew(df['Age'].dropna())}")
print(f"Schiefe von Fare: {skew(df['Fare'].dropna())}")
print(f"Schiefe von FamilySize: {skew(df['FamilySize'].dropna())}")

# Logarithmische Transformation für schiefe Spalten anwenden
df['Fare'] = np.log1p(df['Fare'])
df['Age'] = np.log1p(df['Age'])

# Boxplots anzeigen, um Ausreißer zu visualisieren
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.boxplot(x=df['Age'])
plt.title('Boxplot von Age nach Transformation')
plt.subplot(1, 3, 2)
sns.boxplot(x=df['Fare'])
plt.title('Boxplot von Fare nach Transformation')
plt.subplot(1, 3, 3)
sns.boxplot(x=df['FamilySize'])
plt.title('Boxplot von FamilySize nach Transformation')
plt.tight_layout()
plt.show()

# Ausreißer nach dem Interquartilsabstand (IQR) entfernen
columns_to_check = ['Age', 'Fare', 'FamilySize']
for col in columns_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Deskriptive Statistik der bereinigten Daten anzeigen
print("\nDeskriptive Statistik der bereinigten Daten:")
print(df.describe())

# Zielvariable und Merkmale definieren
X = df.drop('Survived', axis=1)  # Zielspalte entfernen
y = df['Survived']  # Zielvariable

# Daten in Trainings- und Testdaten aufteilen (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell wählen und initialisieren (Logistische Regression)
model = LogisticRegression()

# Modell trainieren
model.fit(X_train, y_train)

# Modellbewertung - Genauigkeit
accuracy = model.score(X_test, y_test)
print(f"\nGenauigkeit: {accuracy:.2f}")

# Confusion Matrix ausgeben
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test, model.predict(X_test)))

# Klassifikationsbericht ausgeben
print("\nKlassifikationsbericht:")
print(classification_report(y_test, model.predict(X_test)))

# Beispiel: Vergleich mit Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_accuracy = rf_model.score(X_test, y_test)
print(f"\nRandom Forest Genauigkeit: {rf_accuracy:.2f}")

# Vergleich der Modelle: Random Forest und Logistische Regression
