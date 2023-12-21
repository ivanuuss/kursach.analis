import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import numpy as np

data = {'Студент': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
    'Экзамен 1': [35, 75, 78, 88, 95, 82, 57, 61, 15, 92, 79, 73, 69, 88, 33, 45, 90, 88, 78, 92, 50, 58],
    'Экзамен 2': [78, 95, 85, 90, 92, 88, 78, 82, 45, 95, 34, 75, 72, 90, 65, 55, 78, 65, 85, 95, 56, 75],
    'Экзамен 3': [92, 88, 90, 85, 78, 92, 85, 78, 66, 88, 92, 65, 64, 85, 28, 65, 71, 78, 90, 88, 63, 52],
    'Экзамен 4': [45, 78, 82, 95, 88, 82, 48, 25, 23, 78, 88, 70, 89, 95, 30, 21, 80, 85, 82, 78, 70, 39],
    'Средний балл': [62.5, 84, 83.75, 89.5, 88.25, 88, 67, 61.5, 37.25, 88.25, 73.25, 70.75, 73.5, 89.5, 39, 46.5, 79.75, 79, 83.75, 88.25, 59.75, 56],
    'Итоговая оценка': [3, 4, 4, 5, 5, 5, 3, 3, 2, 5, 4, 4, 4, 5, 2, 2, 4, 4, 5, 5, 3, 3]
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Статистические тесты (пример t-теста)
group1 = df['Итоговая оценка'] == 4
group2 = df['Итоговая оценка'] == 5
stat, p_value = ttest_ind(df[group1]['Средний балл'], df[group2]['Средний балл'])
print(f'T-статистика: {stat}, p-значение: {p_value}')

# Проверка гипотезы о нормальном распределении
itog_otsenki = df['Итоговая оценка']
stat, p_value = shapiro(itog_otsenki)
# Вывод результатов
print(f"Статистика теста: {stat}")
print(f"P-значение: {p_value}")

# Интерпретация результата
alpha = 0.05
if p_value > alpha:
    print("Не удалось отвергнуть гипотезу о нормальном распределении")
else:
    print("Гипотеза о нормальном распределении отвергнута")

# Корреляционный анализ
correlation_matrix = df.corr()
# Визуализация корреляций
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Матрица корреляций")
plt.show()
# Выводим коэффициенты корреляции в терминал
print("\nКоэффициенты корреляции:")
print(correlation_matrix)

# Доверительные интервалы для корреляций
def correlation_confidence_interval(r, n):
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    z_crit = 1.96  # Для уровня доверия 95%
    ci_lower, ci_upper = np.tanh((z - z_crit*se, z + z_crit*se))
    return ci_lower, ci_upper

confidence_intervals = {}
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2:
            r = correlation_matrix.loc[col1, col2]
            n = len(df)
            confidence_intervals[(col1, col2)] = correlation_confidence_interval(r, n)

# Выводим доверительные интервалы в терминал
print("\nДоверительные интервалы для корреляций:")
for key, value in confidence_intervals.items():
    print(f"{key}: {value}")

# Диаграммы рассеяния для первых 3 переменных
sns.pairplot(df[['Экзамен 1', 'Экзамен 2', 'Экзамен 3']])
plt.show()

# Регрессионный анализ
X = sm.add_constant(df['Экзамен 1'])
y = df['Средний балл']
model = sm.OLS(y, X).fit()
print("")
print(model.summary())

# Выводим регрессионные коэффициенты
print("\nРегрессионные коэффициенты:")
print(model.params)
# Анализ влияния среднего балла
plt.scatter(df['Средний балл'], df['Итоговая оценка'])
plt.xlabel('Средний балл')
plt.ylabel('Итоговая оценка')
plt.show()

# Различия в успеваемости
sns.boxplot(x='Итоговая оценка', y='Средний балл', data=df)
plt.show()

#Применение шагового регрессионного анализа
def stepwise_selection(X, y,
                       initial_list=[],
                       threshold_in=0.01,
                       threshold_out = 0.05,
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # шаг вперед
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # шаг назад
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # используем все переменные для получения p-value
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # найти максимальное p-value
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
result = stepwise_selection(X, y)
# Печать результата
print("Выбранные переменные:")
print(result)

# Множественная регрессия
X = sm.add_constant(df[['Экзамен 1', 'Экзамен 2', 'Экзамен 3', 'Экзамен 4', 'Средний балл']])
y = df['Итоговая оценка']
model = sm.OLS(y, X).fit()
print(model.summary())

# Диаграммы рассеяния для выбранных пар переменных
sns.pairplot(df[['Экзамен 1', 'Экзамен 2', 'Экзамен 3', 'Экзамен 4', 'Средний балл', 'Итоговая оценка']])
plt.show()

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)
# Визуализация результатов
plt.scatter(y_test, y_pred)
plt.xlabel("Фактическая оценка")
plt.ylabel("Предсказанная оценка")
plt.title("Регрессионная модель: Фактическая оценка vs. Предсказанная оценка")
plt.show()

# График фактических и спрогнозированных значений
plt.scatter(df['Студент'], y, label='Фактические значения')
plt.plot(df['Студент'], model.predict(X), label='Спрогнозированные значения', color='red')
plt.xlabel('Студент')
plt.ylabel('Итоговая оценка')
plt.legend()
plt.show()

# Определяем зависимую и независимые переменные
X = df[['Экзамен 1', 'Экзамен 2', 'Экзамен 3', 'Экзамен 4', 'Средний балл']]
# Вычисляем VIF для каждого фактора
vif_data = pd.DataFrame()
vif_data["Фактор"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# Выводим результаты
print(vif_data)

# Разделение данных на обучающий и тестовый наборы
X = df[['Средний балл']]
y = df['Итоговая оценка']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)
# Получение коэффициентов регрессии
intercept = model.intercept_
coef_sredniy_ball = model.coef_[0]
# Прогноз итоговой оценки для каждого студента
df['Прогноз Итоговой оценки'] = intercept + coef_sredniy_ball * df['Средний балл']
# Вывод DataFrame с прогнозами
print(df[['Студент', 'Средний балл', 'Итоговая оценка', 'Прогноз Итоговой оценки']])
