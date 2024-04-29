import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

url = 'netflix_titles.csv'

data = pd.read_csv(url, encoding='latin1')

data.drop(data.columns[data.columns.str.contains('Unnamed')], axis=1, inplace=True)

data = data.dropna(how='any')

def type_cat(x):
    if x == 'TV Show':
        return 0
    elif x == 'Movie':
        return 1
    
def extract_duration(x):
    if pd.isna(x):
        return np.nan
    if 'min' in x:
        return int(x.split(' ')[0])
    elif 'Season' in x:
        return int(x.split(' ')[0]) * 30
    else:
        return np.nan  
    
def categorize_rating(x):
    if isinstance(x, str):
        if x == 'PG-13':
            return 2
        elif x == 'TV-MA':
            return 1
        elif x == 'R':
            return 0
    return -1

data['encoded_rating'] = data['rating'].apply(categorize_rating)
    
data['is_movie'] = data['type'].apply(type_cat)

data['num_duration'] = data.duration.apply(extract_duration)

categories = ["Drama", "Horror", "Comedy", "Action", "Other"] 

def assign_category(listed_in):
    for i, category in enumerate(categories):
        if category.lower() in listed_in.lower():
            return i
    return len(categories)-1

data['category'] = data['listed_in'].apply(assign_category)
y = data['category']
X = data[['is_movie', 'country','release_year', 'num_duration', 'encoded_rating']]

X_encoded = pd.get_dummies(X, columns=['country'])

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_imputed, y_train)

y_pred = linear_model.predict(X_test_imputed)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#převést na přesnost/účinnost - kolik se jich povedlo správně zařadit, tedy v procentech- porovnáme vektor predikce a realita
#a z toho vypočítat predikci
#+ graf x-id y- žánry - v grafu budou dvě linky, jedna trénovací a jedna testovací.

genre_names = categories

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xticks(range(len(categories)), genre_names, rotation=45)
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Predikované hodnoty')
plt.title('Porovnání skutečných hodnot a predikcí')
plt.show()
