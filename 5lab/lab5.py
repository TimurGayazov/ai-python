import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Загрузка данных
file_path = './data/movement_libras_1.data'
df = pd.read_csv(file_path, header=None)

# Масштабирование признаков
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, :-1])  # Масштабируем все столбцы кроме последнего (целевой класс)


# Метод локтя для определения оптимального числа кластеров
def plot_elbow_method(scaled_features):
    distortions = []
    K = range(1, 21)
    for k in K:
        kmean_model = KMeans(n_clusters=k, random_state=42)
        kmean_model.fit(scaled_features)
        distortions.append(kmean_model.inertia_)

    plt.figure(figsize=(12, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Искажение')
    plt.title('Метод локтя для оптимального числа кластеров')
    plt.show()


plot_elbow_method(scaled_features)
# Применение алгоритмов кластеризации
# это методы машинного обучения, которые используются для группировки схожих объектов в наборе данных в отдельные кластеры.
# Целью кластеризации является разделение данных на группы таким образом, чтобы объекты внутри одного кластера были более
# похожи друг на друга, чем на объекты из других кластеров.
#  в медицине — для классификации пациентов на основе симптомов и характеристик заболевания
# Применение алгоритмов кластеризации
algorithms = {
    "KMeans": KMeans(n_clusters=15, random_state=42),  # Указываем 15 кластеров согласно числу классов в данных
    "Affinity Propagation": AffinityPropagation(),
    "Mean Shift": MeanShift() # метод сдвига среднего.
}
# 1)1. выбор центроидов - случайным образом выбирает точка в пространстве данных, которые будут служить начальными центрами кластеров
# 1)2. после каждая точка данных присваивается к ближайшему к ней центроиду, образуя первичный кластер
# 1)3.Центроиды перемещаются в центры своих кластеров, вычисляя среднее значение всех точек в кластере. Повтор пока кластеры не перестанут изменяться

# 2)1.каждая точка данных отправ. сообщения всем другим точкам оценивая их схожесть.
# Схожесть определяется на основе расстояния между точками данных (чем ближе точки, тем они схожее)
# 2)2.получаем: доступность и ответственность. доступность, показывается на сколько точка будет привлекательна как центр кластера,
# а ответственность на сколько сама точка будет предпочитать другую точку в качестве центра
# 2)3.экземпляры кластеров обновляются, чтобы определить наилучшие центры кластеров


#3) Mean Shift работает по следующему принципу: для каждой точки данных вычисляется центр масс всех точек, находящихся в заданном радиусе (окне).
# Затем эта точка сдвигается к вычисленному центру масс. Этот процесс повторяется до тех пор, пока точки не перестанут значительно смещаться.
# В конечном итоге точки, которые сходятся к одному и тому же центру масс, формируют кластер.
#3) aлгоритм кластеризации, который ищет центры кластеров, перемещаясь в направлении наибольшего увеличения плотности точек данных.

results = []

# Обучение и оценка алгоритмов
for name, algorithm in algorithms.items():
    model = algorithm.fit(scaled_features)
    labels = model.labels_
    silhouette_avg = silhouette_score(scaled_features, labels)


    results.append((name, silhouette_avg))
    print(f"{name}:")
    print(f"  Silhouette Score = {silhouette_avg:.4f}") #метрика оценки качества кластеризации, которая помогает определить, насколько хорошо объекты кластеризованы.
    # ближе к 1, хорошо кластеризованы, ближе к 0, указ на нахождение объекта на границе м-ду кластерами, если отриц. то объект неправильно присвоен к кластеру
    # Визуализация кластеров (используем первые два признака для 2D визуализации)
    plt.figure(figsize=(12, 6))
    plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=labels, cmap='viridis')
    plt.title(f"{name} Clustering")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()

# Определение наилучшего алгоритма по silhouette score
best_algorithm = max(results, key=lambda x: x[1])
print(f"\nНаилучший алгоритм по Silhouette Score: {best_algorithm[0]} с Silhouette Score = {best_algorithm[1]:.4f}")

