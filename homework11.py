import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Загрузка и подготовка данных
iris = datasets.load_iris()
X = iris.data[:, :2]  # sepal_length и sepal_width
y = iris.target

# 2. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Построение модели LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 4. Визуализация предсказаний LDA
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=lda.predict(X_test), cmap='viridis', edgecolor='k')
centers = lda.means_
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c=['red', 'green', 'blue'], edgecolors='black', label='Centers') # добавим центры классов
plt.title("LDA Predictions on Test Data")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(*scatter.legend_elements(), title="Classes", loc='upper left')

# 5. Подготовка данных для кластеризации (отбрасываем y)
X_cluster = X.copy()

# 6. Подбор оптимального числа кластеров для KMeans (используем силуэт)
silhouette_scores = []
for n_clusters in range(2, 11):  # проверяем от 2 до 10 кластеров
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    silhouette_avg = silhouette_score(X_cluster, cluster_labels)
    silhouette_scores.append(silhouette_avg)

best_n_clusters = np.argmax(silhouette_scores) + 2 # +2 потому что начинали с 2 кластеров
print(f"Оптимальное количество кластеров: {best_n_clusters}")

# 7. Кластеризация с оптимальным числом кластеров
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_cluster)
centroids = kmeans.cluster_centers_

# 8. Визуализация результатов кластеризации
plt.subplot(1, 2, 2)
scatter_kmeans = plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', edgecolors='black', label='Centroids') # добавляем центроиды
plt.title(f"KMeans Clustering with {best_n_clusters} Clusters")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(*scatter_kmeans.legend_elements(), title="Clusters", loc='upper left') #  legend для кластеров

plt.tight_layout()
plt.show()