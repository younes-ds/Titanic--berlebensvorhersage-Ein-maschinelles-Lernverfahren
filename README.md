# Titanic Überlebensvorhersage

In diesem Projekt geht es darum, das Überleben von Passagieren auf der Titanic vorherzusagen, basierend auf verschiedenen Merkmalen wie Alter, Fahrpreis und Familiengröße. Das Modell verwendet die Logistische Regression und den Random Forest-Algorithmus, um vorherzusagen, ob ein Passagier überlebt hat oder nicht.

**Quelle**: Der Datensatz stammt von Kaggle: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)

## **Ziel**:
Vorhersagen, ob Passagiere der Titanic überlebt haben, basierend auf Merkmalen wie Alter, Fahrpreis, Familiengröße und anderen.

## **Schritte**:
1. **Datenbereinigung**: Fehlende Werte auffüllen, unnötige Spalten entfernen und neue Merkmale erstellen.
2. **Feature Engineering**: Neue Merkmale wie `FamilySize` erstellen und bestehende Merkmale transformieren (logarithmische Transformationen, Skalierung).
3. **Modelltraining**: Modelle mit Logistischer Regression und Random Forest trainieren.
4. **Modellbewertung**: Die Leistung des Modells mit Genauigkeit, Konfusionsmatrix und Klassifikationsbericht bewerten.

## **Merkmale**:
- `Pclass`: Passagierklasse (1., 2. oder 3.)
- `Age`: Alter des Passagiers
- `SibSp`: Anzahl der Geschwister oder Ehepartner an Bord
- `Parch`: Anzahl der Eltern oder Kinder an Bord
- `Fare`: Fahrpreis des Passagiers
- `Sex`: Geschlecht des Passagiers
- `Embarked`: Abfahrtshafen (C, Q, S)

## **Ziel**:
- `Survived`: Ob der Passagier überlebt hat (1) oder nicht (0)

## **Datenvorverarbeitung**:
- Fehlende Werte auffüllen: 
  - `Age`: Mit dem Median auffüllen
  - `Embarked`: Mit dem Modus auffüllen
- Unnötige Spalten entfernen (`Cabin`, `PassengerId`, `Name`, `Ticket`).
- Neues Merkmal erstellen: `FamilySize = SibSp + Parch`.
- Kategorische Variablen mit One-Hot-Encoding bearbeiten: `Sex` und `Embarked`.
- Numerische Merkmale normalisieren: `Age`, `Fare`, `FamilySize`.
- Schiefe Merkmale logarithmisch transformieren: `Age`, `Fare`.

## **Modellierung**:
- **Logistische Regression**: Trainiert mit 80% der Daten und getestet mit 20%.
- **Random Forest**: Trainiert und bewertet mit derselben Datenaufteilung.

## **Bewertung**:
- **Genauigkeit**: Misst die Gesamtleistung des Modells.
- **Konfusionsmatrix**: Zeigt die wahren Positiven, falschen Positiven, wahren Negativen und falschen Negativen.
- **Klassifikationsbericht**: Zeigt Präzision, Recall und F1-Score.

## **Visualisierungen**:
- Boxplots für transformierte Merkmale (`Age`, `Fare`, `FamilySize`).
- Ausreißerentfernung mit der IQR-Methode.

## **Abhängigkeiten**:
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

### **Installation**:
Um die nötigen Abhängigkeiten zu installieren, führen Sie aus:
```bash
pip install -r requirements.txt
