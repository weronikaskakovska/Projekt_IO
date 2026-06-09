# PROJEKT: KLASYFIKACJA GRZYBÓW JADALNYCH I TRUJĄCYCH (AI)

# 1. IMPORTOWANIE BIBLIOTEK
import pandas as pd            # Służy do pracy na tabelach danych (jak Excel)
import numpy as np             # Służy do szybkich obliczeń matematycznych
import matplotlib.pyplot as plt# Służy do tworzenia wykresów
import seaborn as sns          # Służy do rysowania ładniejszych wykresów

# Narzędzia z biblioteki scikit-learn (Sztuczna Inteligencja)
from sklearn.model_selection import train_test_split # Dzieli dane na naukę i test
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Ocena modeli
from sklearn.tree import DecisionTreeClassifier     # Model 1: Drzewo Decyzyjne
from sklearn.neighbors import KNeighborsClassifier # Model 2: K-Najbliższych Sąsiadów
from sklearn.naive_bayes import BernoulliNB         # Model 3: Naiwny Bayes
from sklearn.neural_network import MLPClassifier   # Model 4: Sieć Neuronowa (Wymóg!)

print("START PROJEKTU")

# 1. wczytanie danych i analiza
# Pobieram oficjalny zbiór danych o grzybach bezpośrednio z internetu
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

# Definiuje nazwy kolumn (cech grzybów)
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Wczytuje dane do tabeli Pandas DataFrame
df = pd.read_csv(url, names=column_names)

print("\nPierwsze 3 wiersze z bazy danych:")
print(df.head(3)) # Pokazuje próbkę danych (litery to cechy, np. p=poisonous, e=edible)

# Sprawdzam balans klas (czy liczba jadalnych i trujących jest podobna)
print("\nLiczba grzybów jadalnych (e) i trujących (p):")
print(df['class'].value_counts()) # funkcja biblioteki Pandas, która zlicza wystąpienia unikalnych wartości w danej kolumnie

# Generuje i zapisuje wykres rozkładu klas
plt.figure(figsize=(6, 4))
sns.countplot(x='class', data=df, palette='Set2')
plt.title("Rozkład klas: Jadalne (e) vs Trujące (p)")
plt.savefig('rozklad_klas.png')
plt.close()



# 2. preprocessing (przetwarzanie danych)
# Algorytmy nie rozumieją liter, musimy zamienić tekst na liczby 0 i 1.

# Sprawdzenie czy są puste pola (NaN) - wynik 0 oznacza brak błędów w danych
print("\nLiczba brakujących wartości w bazie:")
print(df.isnull().sum().sum()) # pierwsze .sum() zlicza puste komórki dla każdej kolumny z osobna, drugie .sum() sumuje wyniki w jedną całkowitą liczbę dla całej tabeli

# Podział na X (cechy grzyba) i y (wynik: jadalny czy trujący)
X = df.drop('class', axis=1)
y = df['class'].map({'e': 0, 'p': 1}) # Jadalny = 0, Trujący = 1

# One-Hot Encoding: Zamienia litery na kolumny typu True/False (1 lub 0)
X = pd.get_dummies(X)

# Podział na zbiór treningowy (80% danych do nauki) i testowy (20% danych do sprawdzenia)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDane podzielone: {X_train.shape[0]} do nauki, {X_test.shape[0]} do testu.")


# 3 & 4. klasyfikacja i ewaluacja
# Tworzę słownik na wyniki końcowe, żeby zrobić wykres porównawczy
results = {}

# Funkcja automatyzująca trenowanie i ocenianie
def uruchom_i_ocen_model(nazwa_modelu, model):
    model.fit(X_train, y_train)       # NAUKA: Model analizuje dane treningowe
    y_pred = model.predict(X_test)    # TEST: Model próbuje zgadnąć status nowych grzybów
    
    acc = accuracy_score(y_test, y_pred) # Liczymy ogólną dokładność
    results[nazwa_modelu] = acc * 100
    
    print(f"\n{nazwa_modelu}")
    print(f"Dokładność ogólna: {acc*100:.2f}%") #acc - ogólna dokładność moedlu zapisana jako ulamek, :.2f - 2 liczby po przecinku
    print("\nRaport szczegółowy (Precision, Recall):")
    print(classification_report(y_test, y_pred, target_names=['Jadalny', 'Trujący']))
    print("Macierz pomyłek (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))

# Testuje różne parametry modeli

# Eksperymenty z Drzewem Decyzyjnym (Zmieniamy głębokość drzewa)
uruchom_i_ocen_model("Drzewo Decyzyjne (max_depth=1 - płytkie)", DecisionTreeClassifier(max_depth=1, random_state=42))
uruchom_i_ocen_model("Drzewo Decyzyjne (max_depth=5 - optymalne)", DecisionTreeClassifier(max_depth=5, random_state=42))

# Eksperymenty z KNN (Zmieniamy liczbę sąsiadów)
uruchom_i_ocen_model("KNN (1 sąsiad)", KNeighborsClassifier(n_neighbors=1))
uruchom_i_ocen_model("KNN (5 sąsiadów)", KNeighborsClassifier(n_neighbors=5))

# Test modelu Naiwnego Bayesa
uruchom_i_ocen_model("Naiwny Bayes", BernoulliNB())

# Test Wielowarstwowej Sieci Neuronowej (MLP)
uruchom_i_ocen_model("Sieć Neuronowa (MLP - 10 neuronów)", MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42))


# 5. reguły asocjacyjne / zależności
print("\nREGUŁY ASOCJACYJNE (ANALIZA ZALEŻNOŚCI")
# Szukamy silnych reguł powiązań między zapachem (odor) a klasą grzyba (jadalny/trujący)
# 'n' = brak zapachu, 'f' = zapach zgniły, 'p' = zapach ostry itp.
tabela_zaleznosci = pd.crosstab(df['odor'], df['class'], normalize='index') * 100 #szukam korelacji i tworzę tablicę krzyżową (macierz kontyngencji); normalize='index' - przelicza dane na udziały procentowe wewnątrz każdego wiersza (zapachu)
print("Procentowy udział grzybów jadalnych (e) i trujących (p) w zależności od zapachu:")
print(tabela_zaleznosci)
print("\n[ZALEŻNOŚĆ/REGUŁA]: Jeśli zapach grzyba to 'f' (foul), to w 100% przypadków jest on trujący.")
print("[ZALEŻNOŚĆ/REGUŁA]: Jeśli grzyb nie ma zapachu ('n'), to w 96.6% przypadków jest jadalny.")



# 6. podsumowanie i generowanie wykresu porównawczego
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.values()), y=list(results.keys()), palette="viridis", orient='h')
plt.xlim(50, 105)
plt.title("Porównanie dokładności (Accuracy) modeli AI")
plt.xlabel("Dokładność [%]")
plt.tight_layout()
plt.savefig('porownanie_modeli.png') # Zapisuje wykres końcowy do pliku
plt.close()

print("\nPROJEKT ZAKOŃCZONY SUKCESEM. WYKRESY ZAPISANE")