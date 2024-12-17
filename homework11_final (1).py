import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# --- 1. Загрузка и изучение данных ---
print("Загрузка и изучение данных...")

# Загружаем датасет iris
iris = load_iris()
X = iris.data[:, :2]  # Используем только sepal length и sepal width как признаки.
y = iris.target       # Целевая переменная (классы ирисов).

print(f"Размерность матрицы признаков X: {X.shape}")
print(f"Количество уникальных классов: {len(np.unique(y))}")

# --- 2. Разделение на обучающую и тестовую выборки ---
print("\nРазделение данных на обучающую и тестовую выборки...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Размер обучающей выборки X_train: {X_train.shape}")
print(f"Размер тестовой выборки X_test: {X_test.shape}")


# --- 3. Построение и оценка модели LDA ---
print("\nОбучение модели LDA (Linear Discriminant Analysis)...")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# Вычисляем точность модели LDA
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели LDA: {accuracy:.2f}")

# Проверка достижения критерия (можем поменять логику)
threshold = 0.7  # Можно поменять значение критерия, если необходимо
if accuracy > threshold:
    print(f"Точность модели превышает установленное значение ({threshold}), что является удовлетворительным результатом.")
else:
    print(f"Точность модели не достигает установленного значения ({threshold}). Потребуются дальнейшие исследования.")


# --- 4. Визуализация результатов LDA ---
print("\nВизуализация результатов классификации LDA...")

plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue'] # Задаем цвета для классов
for i, color in enumerate(colors):
    # Истинные классы
    plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1], color=color, label=f'Класс {i} (Истинный)')
    # Предсказанные классы (маркер 'x')
    plt.scatter(X_test[y_pred == i, 0], X_test[y_pred == i, 1], color=color, marker='x', label=f'Класс {i} (Предсказанный)')


# Центры классов (means) на графике
class_centers = lda.means_
plt.scatter(class_centers[:, 0], class_centers[:, 1], color='black', marker='o', s=100, edgecolors='black', label='Центры классов')

plt.title('Классификация ирисов с использованием LDA')
plt.xlabel('Длина чашелистика (Sepal Length)')
plt.ylabel('Ширина чашелистика (Sepal Width)')
plt.legend()
plt.grid(True)
plt.show()

# --- 5. Кластеризация данных с помощью KMeans ---
print("\nКластеризация данных с помощью KMeans...")

num_classes = len(np.unique(y)) # Используем число классов в y для начального количества кластеров
kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)  # Инициализируем KMeans
kmeans.fit(X)  # Обучаем на всех данных X (без разделения)

# Визуализируем результаты кластеризации
print("\nВизуализация результатов кластеризации KMeans...")
plt.figure(figsize=(10, 6))
for i in range(num_classes):
   plt.scatter(X[kmeans.labels_ == i, 0], X[kmeans.labels_ == i, 1], label=f'Кластер {i}')

# Центры кластеров (centroids) на графике
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=100, edgecolors='black', label='Центры кластеров')

plt.title(f'Кластеризация ирисов с использованием KMeans (кол-во кластеров = {num_classes})')
plt.xlabel('Длина чашелистика (Sepal Length)')
plt.ylabel('Ширина чашелистика (Sepal Width)')
plt.legend()
plt.grid(True)
plt.show()


print(f"\nКоличество кластеров: {num_classes}, соответствует числу классов целевой переменной.")
print("\nЗавершение анализа.")