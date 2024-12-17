import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Загрузка данных из текста
data = """
"location","town","mortality","hardness"
"South","Bath",1247,105
"North","Birkenhead",1668,17
"South","Birmingham",1466,5
"North","Blackburn",1800,14
"North","Blackpool",1609,18
"North","Bolton",1558,10
"North","Bootle",1807,15
"South","Bournemouth",1299,78
"North","Bradford",1637,10
"South","Brighton",1359,84
"South","Bristol",1392,73
"North","Burnley",1755,12
"South","Cardiff",1519,21
"South","Coventry",1307,78
"South","Croydon",1254,96
"North","Darlington",1491,20
"North","Derby",1555,39
"North","Doncaster",1428,39
"South","East Ham",1318,122
"South","Exeter",1260,21
"North","Gateshead",1723,44
"North","Grimsby",1379,94
"North","Halifax",1742,8
"North","Huddersfield",1574,9
"North","Hull",1569,91
"South","Ipswich",1096,138
"North","Leeds",1591,16
"South","Leicester",1402,37
"North","Liverpool",1772,15
"North","Manchester",1828,8
"North","Middlesbrough",1704,26
"North","Newcastle",1702,44
"South","Newport",1581,14
"South","Northampton",1309,59
"South","Norwich",1259,133
"North","Nottingham",1427,27
"North","Oldham",1724,6
"South","Oxford",1175,107
"South","Plymouth",1486,5
"South","Portsmouth",1456,90
"North","Preston",1696,6
"South","Reading",1236,101
"North","Rochdale",1711,13
"North","Rotherham",1444,14
"North","St Helens",1591,49
"North","Salford",1987,8
"North","Sheffield",1495,14
"South","Southampton",1369,68
"South","Southend",1257,50
"North","Southport",1587,75
"North","South Shields",1713,71
"North","Stockport",1557,13
"North","Stoke",1640,57
"North","Sunderland",1709,71
"South","Swansea",1625,13
"North","Wallasey",1625,20
"South","Walsall",1527,60
"South","West Bromwich",1627,53
"South","West Ham",1486,122
"South","Wolverhampton",1485,81
"North","York",1378,71
"""

# Преобразование данных в DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), sep=',')

# Задание 1: Анализ по всем данным
print("Анализ для всех данных:")
# 1.1. Построение точечного графика
plt.figure(figsize=(8, 6))
plt.scatter(df['hardness'], df['mortality'])
plt.title('Зависимость смертности от жесткости воды (все города)')
plt.xlabel('Жесткость воды')
plt.ylabel('Смертность')
plt.show()


# 1.2. Расчет коэффициентов корреляции
corr_pearson, _ = pearsonr(df['hardness'], df['mortality'])
corr_spearman, _ = spearmanr(df['hardness'], df['mortality'])
print(f"Коэффициент корреляции Пирсона: {corr_pearson:.2f}")
print(f"Коэффициент корреляции Спирмена: {corr_spearman:.2f}")

# 1.3. Построение модели линейной регрессии
X = df[['hardness']]
y = df['mortality']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


# 1.4. Расчет коэффициента детерминации
r2 = r2_score(y, y_pred)
print(f"Коэффициент детерминации (R^2): {r2:.2f}")

# 1.5. Вывод графика остатков
plt.figure(figsize=(8, 6))
residuals = y - y_pred
plt.scatter(y_pred, residuals)
plt.title('График остатков (все города)')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


# Задание 2: Анализ для северных и южных городов по отдельности
print("\nАнализ для северных и южных городов по отдельности:")
for region in df['location'].unique():
    print(f"\nАнализ для {region} городов:")
    df_region = df[df['location'] == region]

    # 2.1. Построение точечного графика
    plt.figure(figsize=(8, 6))
    plt.scatter(df_region['hardness'], df_region['mortality'])
    plt.title(f'Зависимость смертности от жесткости воды ({region} города)')
    plt.xlabel('Жесткость воды')
    plt.ylabel('Смертность')
    plt.show()

    # 2.2. Расчет коэффициентов корреляции
    corr_pearson, _ = pearsonr(df_region['hardness'], df_region['mortality'])
    corr_spearman, _ = spearmanr(df_region['hardness'], df_region['mortality'])
    print(f"Коэффициент корреляции Пирсона: {corr_pearson:.2f}")
    print(f"Коэффициент корреляции Спирмена: {corr_spearman:.2f}")

    # 2.3. Построение модели линейной регрессии
    X_region = df_region[['hardness']]
    y_region = df_region['mortality']
    model_region = LinearRegression()
    model_region.fit(X_region, y_region)
    y_pred_region = model_region.predict(X_region)

    # 2.4. Расчет коэффициента детерминации
    r2_region = r2_score(y_region, y_pred_region)
    print(f"Коэффициент детерминации (R^2): {r2_region:.2f}")

    # 2.5. Вывод графика остатков
    plt.figure(figsize=(8, 6))
    residuals_region = y_region - y_pred_region
    plt.scatter(y_pred_region, residuals_region)
    plt.title(f'График остатков ({region} города)')
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()