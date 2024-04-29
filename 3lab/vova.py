import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Импорт данных из файла
with open('data.txt', 'r') as file:
    data = file.readlines()

# Создание выборки из данных параметров состояния системы пропульсии
# на основе измеренных признаков и коэффициентов деградации компрессора и турбины.
selection = []
for line in data:
    numbers = list(map(float, line.strip().split()))
    selection.append(numbers)

selection = np.array(selection)

# Разделение выборки на признаки и целевую переменную
X, y = selection[:, :-1], selection[:, -1]

# Разделение выборки на обучающую и тестовую выборки
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Сохранение обучающей выборки в файл
with open('training_set.txt', 'w') as train_file:
    for i in range(len(X_train)):
        train_file.write(' '.join(map(str, np.append(X_train[i], y_train[i]))) + '\n')

# Сохранение тестовой выборки в файл
with open('test_set.txt', 'w') as test_file:
    for i in range(len(X_test)):
        test_file.write(' '.join(map(str, np.append(X_test[i], y_test[i]))) + '\n')

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Построение графика
y_train_pred = model.predict(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, color='blue', label='Predicted vs. True')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--',
         label='Perfect prediction')
plt.title('Predicted vs. True values (Training Set)')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.legend()
plt.grid(True)
plt.show()

# Предсказание значений целевой переменной на тестовой выборке
y_test_pred = model.predict(X_test)

# Оценка точности модели на тестовой выборке
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Точность модели линейной регрессии:")
print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
print(f"Среднеквадратическая ошибка (MSE) на тестовой выборке: {mse_test}")
print(f"Коэффициент детерминации (R-squared) на тестовой выборке: {r2_test}")


# Функция, описывающая истинную зависимость
def true_fun(X):
    return np.cos(1.5 * np.pi * X)


np.random.seed(0)
n_samples = 30
degrees = range(1, 10)
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.rand(n_samples) * 0.1

plt.figure(figsize=(14, 5))

for i, degree in enumerate(degrees, 1):
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([
        ("polynomial_features", polynomial_features),
        ("linear_regression", linear_regression),
    ])
    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
    pipeline.fit(X[:, np.newaxis], y)
    y_pred = pipeline.predict(X[:, np.newaxis])

    # Графики зависимости точности на обучающей и тестовой выборках от степени полиномиальной функции
    plt.subplot(3, 3, i)
    plt.plot(X, y_pred, label="Model")
    plt.plot(X, true_fun(X), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(f"Degree {degree}\nMSE = {-scores.mean():.2e} (+/- {scores.std():.2e})")
plt.show()

# Построение графика зависимости точности на обучающей и тестовой выборках от коэффициента регуляризации
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_accuracy_ridge = []
test_accuracy_ridge = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    train_accuracy_ridge.append(model.score(X_train, y_train))
    test_accuracy_ridge.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(alphas, train_accuracy_ridge, label='Обучающая выборка')
plt.plot(alphas, test_accuracy_ridge, label='Тестовая выборка')
plt.xlabel('Коэффициент регуляризации')
plt.ylabel('Точность')
plt.title('Точность модели с регуляризацией')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()
