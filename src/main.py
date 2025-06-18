import time
import matplotlib
import os

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna

from typing import List, Tuple, Dict, Any, Callable
from numpy.typing import NDArray


# определение типа для функции планировщика скорости обучения
LearningRateScheduler = Callable[[int], float]


def constant_lr(initial_lr: float) -> LearningRateScheduler:
    """постоянный шаг
    формула: α_k = α_0 (где α_k – шаг на k-й итерации, α_0 – начальный шаг).

    преимущества: простота реализации.
    недостатки: сложно подобрать оптимальное значение. слишком большой шаг может привести к расходимости или "перепрыгиванию" через минимум. слишком маленький шаг замедляет сходимость.
    """
    return lambda iteration: initial_lr


def exponential_decay_lr(
    initial_lr: float, decay_rate: float, decay_steps: int
) -> LearningRateScheduler:
    """экспоненциальное затухание шага:
    lr = initial_lr * decay_rate^(iteration / decay_steps)

    преимущества: позволяет более точно настроиться на минимум.
    недостатки: требует подбора дополнительных гиперпараметров (начальный шаг, скорость затухания).
    """
    return lambda iteration: initial_lr * (decay_rate ** (iteration / decay_steps))


def step_decay_lr(
    initial_lr: float, drop_rate: float, epochs_drop: int
) -> LearningRateScheduler:
    """ступенчатое уменьшение шага

    шаг остается постоянным в течение определенного числа эпох, а затем уменьшается на некоторый коэффициент.
    преимущества: дает алгоритму время "устояться" с текущим шагом перед его уменьшением.
    недостатки: требует подбора начального шага, коэффициента уменьшения и частоты уменьшения.
    """
    return lambda iteration: initial_lr * drop_rate ** np.floor(
        (1 + iteration) / epochs_drop
    )


def inverse_time_decay_lr(initial_lr: float, decay_rate: float) -> LearningRateScheduler:
    """обратно-временное затухание шага
    формула: lr = initial_lr / (1 + decay_rate * iteration)

    преимущества: обеспечивает плавное уменьшение шага с течением времени.
    недостатки: может быть слишком медленным для некоторых задач.
    """
    return lambda iteration: initial_lr / (1 + decay_rate * iteration)


class SGDRegressorManual:
    """
    самостоятельная реализация стохастического градиентного спуска (SGD) для линейной регрессии.
    этот класс не использует внешние библиотеки для оптимизации, чтобы продемонстрировать
    основные принципы работы алгоритма.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        lr_decay: float = 0.0,
        lr_scheduler: LearningRateScheduler = None,
        random_state: int = 42,
    ) -> None:
        """
        инициализация регрессора.

        параметры:
        - learning_rate (float): скорость обучения (шаг градиентного спуска), η. определяет, насколько
          сильно мы изменяем веса на каждой итерации.
        - n_epochs (int): количество полных проходов по всему обучающему набору данных.
        - batch_size (int): размер мини-выборки (батча). если равен 1, это "чистый" SGD.
          если равен размеру всей выборки, это классический градиентный спуск (GD).
        - l1_reg (float): коэффициент L1-регуляризации (Lasso), λ₁. добавляет в функцию потерь
          штраф, пропорциональный сумме абсолютных значений весов. способствует обнулению
          незначимых весов, производя отбор признаков.
        - l2_reg (float): коэффициент L2-регуляризации (Ridge), λ₂. добавляет штраф,
          пропорциональный сумме квадратов весов. препятствует слишком большим значениям весов,
          делая модель более устойчивой.
        - lr_decay (float): коэффициент затухания скорости обучения. позволяет уменьшать шаг
          с каждой эпохой, что помогает алгоритму сойтись точнее. используется для обратно-временного затухания.
        - lr_scheduler (LearningRateScheduler): функция планировщика скорости обучения.
          если задана, переопределяет поведение lr_decay.
        - random_state (int): зерно для генератора случайных чисел для воспроизводимости результатов.
        """
        """
        признаки: количесто_комнат, площадь, возраст, расстояние_до_метро,
        batch_size = 3
        [
        1. 2, 10, 15, 10
        2. 1, 7, 10, 10
        3. 1, 7, 10, 10
        ]
        [
        4. 1, 7, 10, 10
        5. 1, 7, 10, 10
        6. 1, 7, 10, 10
        ]

        цена (y)
        1. 1000000
        2. 500000
        3. 500000
        4. 500000
        5. 500000
        6. 500000
        
        """
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.lr_decay = lr_decay
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is None and lr_decay > 0:
            # если планировщик не задан, но задан коэффициент затухания,
            # используем обратно-временное затухание (старое поведение)
            self.lr_scheduler = inverse_time_decay_lr(learning_rate, lr_decay)
        elif lr_scheduler is None:
            # если ни планировщик, ни коэффициент затухания не заданы,
            # используем постоянную скорость обучения
            self.lr_scheduler = constant_lr(learning_rate)
        self.random = np.random.RandomState(random_state)
        self.weights: NDArray[np.float64] | None = None
        self.loss_history: List[float] = []
        self.lr_history: List[float] = []  # добавляем историю изменения скорости обучения

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> "SGDRegressorManual":
        """
        обучение модели на данных.

        математика:
        1. модель линейной регрессии: y_pred = X @ w
        2. функция потерь (MSE - Mean Squared Error): L = (1/N) * Σ(y_predᵢ - yᵢ)²
        3. градиент MSE по весам w: ∇L_w = (2/N) * Xᵀ @ (y_pred - y)
        4. штраф L1 (Lasso): λ₁ * Σ|wᵢ|. его субградиент: λ₁ * sign(w)
        5. штраф L2 (Ridge): λ₂ * Σwᵢ². его градиент: 2 * λ₂ * w
        6. обновление весов (шаг градиентного спуска): w = w - η * ∇L_total
           где η - скорость обучения, а ∇L_total - градиент с учетом регуляризации.
        """
        # "стохастический" — это синоним слова "случайный". он так называется, потому что на каждом шаге мы выбираем одну случайную точку данных из выборки и делаем вид, что градиент, посчитанный только по ней, — это градиент для всей выборки.
        n_samples, n_features = X.shape
        # инициализируем веса случайными значениями. n_features включает и фиктивный признак для смещения (bias).
        self.weights = self.random.randn(n_features, 1)

        for epoch in range(self.n_epochs):
            # обновление скорости обучения с помощью планировщика
            self.lr = self.lr_scheduler(epoch)
            self.lr_history.append(self.lr)

            # перемешивание данных в начале каждой эпохи - ключевой аспект стохастических методов.
            # это помогает избежать циклов и улучшает сходимость.
            indices = np.arange(n_samples)
            self.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = []
            # итерация по мини-батчам
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                # 1. предсказание модели на текущем батче
                y_pred = X_batch @ self.weights
                # 2. вычисление ошибки
                error = y_pred - y_batch

                # 3. вычисление градиентов от регуляризаций
                l1_grad = self.l1_reg * np.sign(self.weights)
                l2_grad = self.l2_reg * 2 * self.weights
                # 4. вычисление градиента функции потерь MSE и сложение с градиентами регуляризаций
                gradient = (
                    (2 / len(X_batch)) * X_batch.T @ error + l1_grad + l2_grad
                )

                # 5. обновление весов - шаг в направлении анти-градиента
                self.weights -= self.lr * gradient

                # сохраняем значение функции потерь (без регуляризации) для истории
                loss = np.mean(error**2)
                epoch_loss.append(loss)

            self.loss_history.append(np.mean(epoch_loss))
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """делает предсказания с помощью обученной модели."""
        if self.weights is None:
            raise ValueError("модель еще не обучена. вызовите fit() перед предсказанием.")
        # предсказание - это просто матричное произведение входных признаков на найденные веса.
        return X @ self.weights


class AdamManual:
    """
    самостоятельная реализация оптимизатора Adam для задачи регрессии.
    Adam (Adaptive Moment Estimation) - это адаптивный метод оптимизации, который вычисляет
    индивидуальные скорости обучения для каждого параметра.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        random_state: int = 42,
    ) -> None:
        """
        инициализация оптимизатора.

        параметры:
        - beta1 (float): коэффициент затухания для первого момента (скользящее среднее градиентов).
        - beta2 (float): коэффициент затухания для второго момента (скользящее среднее квадратов градиентов).
        - epsilon (float): малая константа для предотвращения деления на ноль.
        """
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.random = np.random.RandomState(random_state)
        self.weights: NDArray[np.float64] | None = None
        self.loss_history: List[float] = []
        # 'm' - первый момент (импульс, momentum), 'v' - второй момент (адаптивная часть)
        self.m: NDArray[np.float64] = np.array(0)
        self.v: NDArray[np.float64] = np.array(0)
        self.t: int = 0  # счетчик шагов (итераций)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "AdamManual":
        """
        обучение модели с использованием Adam.

        математика Adam:
        на каждом шаге t:
        1. вычислить градиент: gₜ = ∇L(wₜ₋₁)
        2. обновить первый момент (экспоненциальное скользящее среднее градиентов):
           mₜ = β₁ * mₜ₋₁ + (1 - β₁) * gₜ
        3. обновить второй момент (экспоненциальное скользящее среднее квадратов градиентов):
           vₜ = β₂ * vₜ₋₁ + (1 - β₂) * gₜ²
        4. скорректировать смещение моментов (важно на первых итерациях):
           m̂ₜ = mₜ / (1 - β₁ᵗ)
           v̂ₜ = vₜ / (1 - β₂ᵗ)
        5. обновить веса:
           wₜ = wₜ₋₁ - η * m̂ₜ / (√v̂ₜ + ε)
        """
        n_samples, n_features = X.shape
        self.weights = self.random.randn(n_features, 1)

        # инициализация векторов моментов нулями
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0

        for epoch in range(self.n_epochs):
            indices = np.arange(n_samples)
            self.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = []
            for i in range(0, n_samples, self.batch_size):
                self.t += 1  # увеличиваем счетчик шагов
                X_batch = X_shuffled[i : i + self.batch_size]
                y_batch = y_shuffled[i : i + self.batch_size]

                if self.weights is None:
                    continue

                y_pred = X_batch @ self.weights
                error = y_pred - y_batch

                # 1. вычисление градиента
                gradient = (2 / len(X_batch)) * X_batch.T @ error

                # 2. обновление первого момента
                self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
                # 3. обновление второго момента
                self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

                # 4. коррекция смещения
                m_hat = self.m / (1 - self.beta1**self.t)
                v_hat = self.v / (1 - self.beta2**self.t)

                # 5. обновление весов
                self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                loss = np.mean(error**2)
                epoch_loss.append(loss)

            self.loss_history.append(np.mean(epoch_loss))
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """делает предсказания с помощью обученной модели."""
        if self.weights is None:
            raise ValueError("модель еще не обучена. вызовите fit() перед предсказанием.")
        return X @ self.weights


def objective_sgd(
    trial: optuna.Trial, X: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """целевая функция для Optuna для подбора гиперпараметров SGDRegressorManual."""
    # определяем пространство поиска гиперпараметров
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    l1_reg = trial.suggest_float("l1_reg", 1e-5, 1.0, log=True)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1.0, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["constant", "exponential", "step", "inverse"]
    )

    # параметры для планировщиков
    if scheduler_type == "exponential":
        decay_rate = trial.suggest_float("decay_rate", 0.8, 1.0)
        decay_steps = trial.suggest_int("decay_steps", 1, 10)
        scheduler = exponential_decay_lr(
            initial_lr=lr, decay_rate=decay_rate, decay_steps=decay_steps
        )
    elif scheduler_type == "step":
        drop_rate = trial.suggest_float("drop_rate", 0.1, 0.9)
        epochs_drop = trial.suggest_int("epochs_drop", 5, 20)
        scheduler = step_decay_lr(
            initial_lr=lr, drop_rate=drop_rate, epochs_drop=epochs_drop
        )
    elif scheduler_type == "inverse":
        decay_rate = trial.suggest_float("decay_rate", 1e-3, 1.0)
        scheduler = inverse_time_decay_lr(initial_lr=lr, decay_rate=decay_rate)
    else:  # constant
        scheduler = constant_lr(initial_lr=lr)

    # кросс-валидация для более надежной оценки
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = SGDRegressorManual(
            learning_rate=lr,
            n_epochs=50,  # фиксированное число эпох для тюнинга
            batch_size=batch_size,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            lr_scheduler=scheduler,
            random_state=42,
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_pred)
        scores.append(mse)

    return np.mean(scores)


def objective_adam(
    trial: optuna.Trial, X: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """целевая функция для Optuna для подбора гиперпараметров AdamManual."""
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    beta1 = trial.suggest_float("beta1", 0.8, 0.999)
    beta2 = trial.suggest_float("beta2", 0.9, 0.9999)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = AdamManual(
            learning_rate=lr,
            n_epochs=50,
            batch_size=batch_size,
            beta1=beta1,
            beta2=beta2,
            random_state=42,
        )
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_pred)
        scores.append(mse)

    return np.mean(scores)


def run_optuna_studies(
    X_train: NDArray[np.float64], y_train: NDArray[np.float64], n_trials: int = 50
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """запускает исследования Optuna для SGD и Adam, сохраняет результаты."""
    # --- исследование для SGD ---
    print("--- Запуск Optuna для SGD ---")
    study_sgd = optuna.create_study(direction="minimize")
    study_sgd.optimize(lambda trial: objective_sgd(trial, X_train, y_train), n_trials=n_trials)

    print(f"Лучшие параметры для SGD: {study_sgd.best_params}")
    df_sgd_study = study_sgd.trials_dataframe()
    df_sgd_study.to_csv("results/optuna_sgd_study.csv", index=False)
    print("Результаты исследования SGD сохранены в results/optuna_sgd_study.csv")

    # --- исследование для Adam ---
    print("\n--- Запуск Optuna для Adam ---")
    study_adam = optuna.create_study(direction="minimize")
    study_adam.optimize(lambda trial: objective_adam(trial, X_train, y_train), n_trials=n_trials)

    print(f"Лучшие параметры для Adam: {study_adam.best_params}")
    df_adam_study = study_adam.trials_dataframe()
    df_adam_study.to_csv("results/optuna_adam_study.csv", index=False)
    print("Результаты исследования Adam сохранены в results/optuna_adam_study.csv")
    
    return study_sgd.best_params, study_adam.best_params


"""

"""
def generate_data(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: float = 20,
    random_state: int = 42,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    генерирует и подготавливает данные для задачи регрессии.
    """
    # создаем синтетический набор данных для линейной регрессии.
    # coef=True возвращает истинные коэффициенты, с которыми были сгенерированы данные.
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        coef=True,
        random_state=random_state,
    )
    # используется функция make_regression из scikit-learn, которая создает данные, где целевая переменная линейно зависит от признаков с некоторым шумом.

    # приводим y к вектор-столбцу для удобства матричных операций
    y = y.reshape(-1, 1)

    # разделяем данные на обучающую и тестовую выборки 80 - 20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # масштабирование признаков - критически важный шаг для градиентных методов.
    # он приводит все признаки к одному масштабу (здесь - среднее 0, стд. отклонение 1),
    # что делает "ландшафт" функции потерь более равномерным и ускоряет сходимость.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    """
    признаки приводятся к стандартному масштабу (среднее 0, стандартное отклонение 1), что очень важно для сходимости градиентных методов.

    """

    # добавляем фиктивный признак (столбец из единиц) к данным.
    # это позволяет модели выучить смещение (intercept, или β₀ в уравнении y = β₀ + β₁x₁ + ...).
    # вес, соответствующий этому признаку, и будет являться смещением.
    X_train_final = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
    X_test_final = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

    return X_train_final, X_test_final, y_train, y_test, coef


class TorchLR(nn.Module):
    """
    простая модель линейной регрессии, реализованная на PyTorch.
    nn.Module - базовый класс для всех нейросетевых модулей в PyTorch.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        # определяем один линейный слой. он будет принимать n_features входов
        # и производить 1 выход (предсказанное значение).
        # этот слой и реализует операцию X @ w + b.
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """определяет прямой проход данных через модель."""
        return self.linear(x)


def train_torch_model(
    optimizer_class: Any,  # здесь может быть любой оптимизатор из torch.optim
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    lr: float = 0.01,
    n_epochs: int = 50,
    **optimizer_params,
) -> Tuple[float, List[float]]:
    """
    обучает модель PyTorch с заданным классом оптимизатора и возвращает
    финальное значение MSE и историю потерь.
    """
    # PyTorch работает со своими структурами данных - тензорами.
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # DataLoader - удобный инструмент PyTorch для автоматического подачи данных
    # батчами и их перемешивания.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # инициализация модели, оптимизатора и функции потерь
    model = TorchLR(X_train.shape[1])
    # создаем экземпляр оптимизатора, передавая ему параметры модели и скорость обучения
    # Создаем экземпляр оптимизатора, передавая ему параметры модели и скорость обучения
    optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_params)
    # MSELoss - это реализация среднеквадратичной ошибки в PyTorch
    criterion = nn.MSELoss()
    loss_history: List[float] = []

    model.train()
    for epoch in range(n_epochs):
        epoch_loss: List[float] = []
        # Стандартный цикл обучения в PyTorch
        for X_batch, y_batch in train_loader:
            # 1. Обнуляем градиенты с предыдущего шага
            optimizer.zero_grad()
            # 2. Делаем предсказание (прямой проход)
            y_pred = model(X_batch)
            # 3. Считаем ошибку
            loss = criterion(y_pred, y_batch)
            # 4. Вычисляем градиенты (обратный проход)
            loss.backward()
            # 5. Обновляем веса
            optimizer.step()
            epoch_loss.append(loss.item())
        loss_history.append(np.mean(epoch_loss))

    # Оценка модели на тестовых данных
    model.eval()
    with torch.no_grad():  # Отключаем вычисление градиентов для ускорения
        y_pred_test = model(X_test_tensor)
        final_mse = criterion(y_pred_test, y_test_tensor).item()

    return final_mse, loss_history


def main() -> None:
    """
    Основная функция, которая запускает все эксперименты:
    - Генерирует данные
    - Обучает самописные и библиотечные модели
    - Собирает результаты
    - Строит графики для анализа
    """
    # --- 0. Создание папки для результатов ---
    os.makedirs("results", exist_ok=True)

    # --- 1. Подготовка данных ---
    X_train, X_test, y_train, y_test, _ = generate_data()

    # --- 1.5. Подбор гиперпараметров с помощью Optuna ---
    # Уменьшим количество попыток для скорости, для реального исследования стоит увеличить до 100+
    best_sgd_params, best_adam_params = run_optuna_studies(X_train, y_train, n_trials=50)

    # --- 2. Обучение моделей и сбор результатов ---
    results: Dict[str, Any] = {
        "Model": [],
        "MSE": [],
        "Time (s)": [],
        "Loss History": [],
        "LR History": [],  # Добавляем историю изменения скорости обучения
    }

    # Самописный SGD с обратно-временным затуханием (старый способ через lr_decay)
    model_manual_sgd = SGDRegressorManual(n_epochs=50, lr_decay=0.1)
    start_time = time.time()
    model_manual_sgd.fit(X_train, y_train)
    duration = time.time() - start_time
    y_pred_manual_sgd = model_manual_sgd.predict(X_test)
    mse_manual_sgd = mean_squared_error(y_test, y_pred_manual_sgd)

    results["Model"].append("SGD (Inverse Time Decay)")
    results["MSE"].append(mse_manual_sgd)
    results["Time (s)"].append(duration)
    results["Loss History"].append(model_manual_sgd.loss_history)
    results["LR History"].append(model_manual_sgd.lr_history)

    # Самописный SGD с экспоненциальным затуханием
    exp_scheduler = exponential_decay_lr(initial_lr=0.01, decay_rate=0.95, decay_steps=1)
    model_exp_decay = SGDRegressorManual(n_epochs=50, lr_scheduler=exp_scheduler)
    start_time = time.time()
    model_exp_decay.fit(X_train, y_train)
    duration = time.time() - start_time
    y_pred_exp_decay = model_exp_decay.predict(X_test)
    mse_exp_decay = mean_squared_error(y_test, y_pred_exp_decay)

    results["Model"].append("SGD (Exponential Decay)")
    results["MSE"].append(mse_exp_decay)
    results["Time (s)"].append(duration)
    results["Loss History"].append(model_exp_decay.loss_history)
    results["LR History"].append(model_exp_decay.lr_history)

    # Самописный SGD со ступенчатым затуханием
    step_scheduler = step_decay_lr(initial_lr=0.01, drop_rate=0.5, epochs_drop=10)
    model_step_decay = SGDRegressorManual(n_epochs=50, lr_scheduler=step_scheduler)
    start_time = time.time()
    model_step_decay.fit(X_train, y_train)
    duration = time.time() - start_time
    y_pred_step_decay = model_step_decay.predict(X_test)
    mse_step_decay = mean_squared_error(y_test, y_pred_step_decay)

    results["Model"].append("SGD (Step Decay)")
    results["MSE"].append(mse_step_decay)
    results["Time (s)"].append(duration)
    results["Loss History"].append(model_step_decay.loss_history)
    results["LR History"].append(model_step_decay.lr_history)

    # Самописный SGD с L1 регуляризацией (Lasso)
    model_manual_l1 = SGDRegressorManual(n_epochs=50, l1_reg=0.1, l2_reg=0.0)
    start_time = time.time()
    model_manual_l1.fit(X_train, y_train)
    duration_l1 = time.time() - start_time
    y_pred_manual_l1 = model_manual_l1.predict(X_test)
    mse_manual_l1 = mean_squared_error(y_test, y_pred_manual_l1)

    results["Model"].append("SGD (L1/Lasso)")
    results["MSE"].append(mse_manual_l1)
    results["Time (s)"].append(duration_l1)
    results["Loss History"].append(model_manual_l1.loss_history)
    results["LR History"].append(model_manual_l1.lr_history)

    # Самописный SGD с L2 регуляризацией (Ridge)
    model_manual_l2 = SGDRegressorManual(n_epochs=50, l1_reg=0.0, l2_reg=0.1)
    start_time = time.time()
    model_manual_l2.fit(X_train, y_train)
    duration_l2 = time.time() - start_time
    y_pred_manual_l2 = model_manual_l2.predict(X_test)
    mse_manual_l2 = mean_squared_error(y_test, y_pred_manual_l2)

    results["Model"].append("SGD (L2/Ridge)")
    results["MSE"].append(mse_manual_l2)
    results["Time (s)"].append(duration_l2)
    results["Loss History"].append(model_manual_l2.loss_history)
    results["LR History"].append(model_manual_l2.lr_history)

    # Самописный SGD с L1/L2 регуляризацией (Elastic Net)
    model_manual_reg = SGDRegressorManual(n_epochs=50, l1_reg=0.05, l2_reg=0.05)
    start_time = time.time()
    model_manual_reg.fit(X_train, y_train)
    duration_reg = time.time() - start_time
    y_pred_manual_reg = model_manual_reg.predict(X_test)
    mse_manual_reg = mean_squared_error(y_test, y_pred_manual_reg)

    results["Model"].append("SGD (Elastic Net)")
    results["MSE"].append(mse_manual_reg)
    results["Time (s)"].append(duration_reg)
    results["Loss History"].append(model_manual_reg.loss_history)
    results["LR History"].append(model_manual_reg.lr_history)

    # Самописный Adam
    model_manual_adam = AdamManual(n_epochs=50)
    start_time = time.time()
    model_manual_adam.fit(X_train, y_train)
    duration_adam = time.time() - start_time
    y_pred_manual_adam = model_manual_adam.predict(X_test)
    mse_manual_adam = mean_squared_error(y_test, y_pred_manual_adam)

    results["Model"].append("Adam")
    results["MSE"].append(mse_manual_adam)
    results["Time (s)"].append(duration_adam)
    results["Loss History"].append(model_manual_adam.loss_history)
    results["LR History"].append([0] * 50)  # Заглушка для Adam, т.к. там нет lr_history

    # --- Новые модели с оптимальными параметрами ---

    # SGD с лучшими параметрами от Optuna
    # Воссоздаем планировщик на основе лучших параметров
    lr_opt = best_sgd_params['learning_rate']
    scheduler_type_opt = best_sgd_params['scheduler_type']
    
    if scheduler_type_opt == "exponential":
        scheduler_opt = exponential_decay_lr(
            initial_lr=lr_opt, 
            decay_rate=best_sgd_params['decay_rate'], 
            decay_steps=best_sgd_params['decay_steps']
        )
    elif scheduler_type_opt == "step":
        scheduler_opt = step_decay_lr(
            initial_lr=lr_opt, 
            drop_rate=best_sgd_params['drop_rate'], 
            epochs_drop=best_sgd_params['epochs_drop']
        )
    elif scheduler_type_opt == "inverse":
        scheduler_opt = inverse_time_decay_lr(
            initial_lr=lr_opt, 
            decay_rate=best_sgd_params['decay_rate']
        )
    else:  # constant
        scheduler_opt = constant_lr(initial_lr=lr_opt)

    model_sgd_optuna = SGDRegressorManual(
        n_epochs=50, 
        learning_rate=lr_opt,
        l1_reg=best_sgd_params['l1_reg'],
        l2_reg=best_sgd_params['l2_reg'],
        batch_size=best_sgd_params['batch_size'],
        lr_scheduler=scheduler_opt
    )
    start_time = time.time()
    model_sgd_optuna.fit(X_train, y_train)
    duration = time.time() - start_time
    y_pred_sgd_optuna = model_sgd_optuna.predict(X_test)
    mse_sgd_optuna = mean_squared_error(y_test, y_pred_sgd_optuna)

    results["Model"].append("SGD (Optuna Best)")
    results["MSE"].append(mse_sgd_optuna)
    results["Time (s)"].append(duration)
    results["Loss History"].append(model_sgd_optuna.loss_history)
    results["LR History"].append(model_sgd_optuna.lr_history)


    # Adam с лучшими параметрами от Optuna
    model_adam_optuna = AdamManual(
        n_epochs=50, 
        learning_rate=best_adam_params['learning_rate'],
        beta1=best_adam_params['beta1'],
        beta2=best_adam_params['beta2'],
        batch_size=best_adam_params['batch_size']
    )
    start_time = time.time()
    model_adam_optuna.fit(X_train, y_train)
    duration = time.time() - start_time
    y_pred_adam_optuna = model_adam_optuna.predict(X_test)
    mse_adam_optuna = mean_squared_error(y_test, y_pred_adam_optuna)

    results["Model"].append("Adam (Optuna Best)")
    results["MSE"].append(mse_adam_optuna)
    results["Time (s)"].append(duration)
    results["Loss History"].append(model_adam_optuna.loss_history)
    results["LR History"].append([0] * 50) # Заглушка

    # Модели из PyTorch
    torch_optimizers = {
        "Torch SGD": (optim.SGD, {}),
        "Torch SGD+Momentum": (optim.SGD, {"momentum": 0.9}),
        "Torch Adagrad": (optim.Adagrad, {}),
        "Torch RMSprop": (optim.RMSprop, {}),
        "Torch Adam": (optim.Adam, {}),
        "Torch SGD+Nesterov": (optim.SGD, {"momentum": 0.9, "nesterov": True}),
    }

    for name, (opt_class, opt_params) in torch_optimizers.items():
        start_time = time.time()
        mse, loss_hist = train_torch_model(
            opt_class,
            X_train,
            y_train,
            X_test,
            y_test,
            **opt_params
        )
        duration = time.time() - start_time
        results["Model"].append(name)
        results["MSE"].append(mse)
        results["Time (s)"].append(duration)
        results["Loss History"].append(loss_hist)
        results["LR History"].append([0] * len(loss_hist))  # Заглушка для PyTorch оптимизаторов

    # --- 3. Анализ и визуализация результатов ---
    df_results = pd.DataFrame(results)
    print("--- Сводные результаты ---")
    print(df_results.drop(["Loss History", "LR History"], axis=1).round(4))

    # Построение графиков кривых обучения
    plt.figure(figsize=(12, 8))
    for i, row in df_results.iterrows():
        plt.plot(row["Loss History"], label=row["Model"])

    plt.title("Динамика функции потерь (MSE) по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")  # Логарифмическая шкала для лучшей визуализации
    plt.tight_layout()
    plt.savefig("results/loss_curves.png", dpi=300)
    plt.close()
    print("\nГрафик кривых обучения сохранен в results/loss_curves.png")

    # Построение графиков изменения скорости обучения
    plt.figure(figsize=(12, 6))
    # Отбираем только модели с ручной реализацией планировщиков
    lr_schedulers = df_results[df_results["Model"].isin([
        "SGD (Inverse Time Decay)", 
        "SGD (Exponential Decay)", 
        "SGD (Step Decay)"
    ])]
    
    for i, row in lr_schedulers.iterrows():
        plt.plot(row["LR History"], label=row["Model"])

    plt.title("Стратегии изменения скорости обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/lr_schedulers.png", dpi=300)
    plt.close()
    print("График стратегий изменения скорости обучения сохранен в results/lr_schedulers.png")

    # Сравнение эффективности разных стратегий затухания LR
    plt.figure(figsize=(12, 6))
    lr_comparison = df_results[df_results["Model"].isin([
        "SGD (Inverse Time Decay)", 
        "SGD (Exponential Decay)", 
        "SGD (Step Decay)"
    ])]
    
    for i, row in lr_comparison.iterrows():
        plt.plot(row["Loss History"], label=f"{row['Model']} (MSE={row['MSE']:.2f})")

    plt.title("Сравнение стратегий затухания скорости обучения")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("results/lr_strategies_comparison.png", dpi=300)
    plt.close()
    print("График сравнения стратегий затухания сохранен в results/lr_strategies_comparison.png")

    # Построение столбчатых диаграмм для сравнения MSE и времени
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Сортировка для наглядности
    df_sorted_mse = df_results.sort_values("MSE", ascending=False)
    sns.barplot(x="MSE", y="Model", data=df_sorted_mse, ax=axes[0], palette="viridis")
    axes[0].set_title("Сравнение моделей по финальному MSE")
    axes[0].set_xlabel("Mean Squared Error (на тестовой выборке)")
    axes[0].set_ylabel("")

    df_sorted_time = df_results.sort_values("Time (s)", ascending=False)
    sns.barplot(
        x="Time (s)", y="Model", data=df_sorted_time, ax=axes[1], palette="plasma"
    )
    axes[1].set_title("Сравнение моделей по времени обучения")
    axes[1].set_xlabel("Время (секунды)")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("results/comparison.png", dpi=300)
    plt.close()
    print("График сравнения моделей сохранен в results/comparison.png")


if __name__ == "__main__":
    main()
