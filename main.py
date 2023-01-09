import math
import numpy as np
from typing import Tuple, List, Callable
import matplotlib.pyplot as plt


# Интерполяционный многочлен в форме Лагранжа
def Q(grid: List[float], i: int) -> Callable[[float], float]:  # возвращает функцию Q_n,i
    def q(x: float) -> float:
        res = 1
        for j, node in enumerate(grid):
            if j != i:
                res *= (x - node)/(grid[i] - node)
        return res
    return q


def interpolate_poly(x_all: List[float], y: List[float]) -> Callable[[float], float]:  # Возвращает L_n
    all_func_q = [Q(x_all, i) for i in range(len(x_all))]

    def inter_func(x: float) -> float:
        res = 0
        for i in range(len(x_all)):
            res += y[i] * all_func_q[i](x)
        return res
    return inter_func
# Конец


def mse_built_matrix(x_all: List[float], y: List[float], n: int) -> Tuple[List[List[float]], List[float]]:
    tmp_matrix = []
    for i in range(n + 1):
        tmp_matrix.append([x ** i for x in x_all])
    A = []
    for i in range(n + 1):
        A.append([0] * (n + 1))
    for i in range(n + 1):  # Строим матрицу А для решения СЛАУ
        for j in range(n + 1):
            A[i][j] = sum([tmp_matrix[i][t] * tmp_matrix[j][t] for t in range(len(x_all))])
    f = [sum([y[j] * tmp_matrix[i][j] for j in range(len(x_all))]) for i in range(n + 1)]
    return A, f


def gauss(A: List[List[float]], f: List[float]) -> List[float]:
    n = len(A)
    for i in range(n - 1):
        for j in range(i + 1, n):
            f[j] -= f[i] * (A[j][i] / A[i][i])
            for k in range(i + 1, n):
                A[j][k] -= A[i][k] * (A[j][i] / A[i][i])
            A[j][i] = 0
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (f[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x


def approx(x_all: List[float], y: List[float], n: int) -> Callable[[float], float]:
    A, f = mse_built_matrix(x_all, y, n)
    coefficients = gauss(A, f)

    def approx_func(x: float) -> float:
        return sum([coefficients[i] * (x ** i) for i in range(n + 1)])
    return approx_func


def read_nodes(file_name: str = 'inp.txt') -> Tuple[List[float], List[float], int]:  # Считать массив значений из файла
    f = open(file_name, 'r')
    x, y, n = [], [], 0
    for i, line in enumerate(f):
        if i == 0:
            n = int(line[:-1])
        else:
            a, b = tuple(map(float, line[:-1].replace(',', '.').split()))
            x.append(a)
            y.append(b)
    f.close()
    return x, y, n


def get_delta(f: Callable[[float], float], g: Callable[[float], float], num_points: int = 400) -> float:  # Вычисление "разности" между f и g
    grid, dx = np.linspace(0, 2, num_points, retstep=True)
    return math.sqrt(sum([((f(x) - g(x))**2) * dx for x in grid]))


x, y, n = read_nodes('inp.txt')
Interp_Lagrange = interpolate_poly(x, y)
Approx = approx(x, y, n)

plt.scatter(x, y, label='True values')
x_grid = np.linspace(x[0], x[-1], 100)
plt.plot(x_grid, [Interp_Lagrange(x) for x in x_grid], label='Interpolate func')
plt.plot(x_grid, [Approx(x) for x in x_grid], label='Approximating func')
plt.legend()
plt.show()

# Здесь вывожу функции, построенные на зашумлённой сетке
'''s = 0.1 
d = np.random.normal(0, 1, len(x))
y_noisy = [y[i] + d[i] * s for i in range(len(y))]
inter = interpolate_poly(x, y_noisy)
appr = approx(x, y_noisy, n)

print(get_delta(inter, Interp_Lagrange))
print(get_delta(appr, Approx))

plt.scatter(x, y, label='True values')
x_grid = np.linspace(x[0], x[-1], 100)
plt.plot(x_grid, [inter(x) for x in x_grid], label='Interpolate func')
plt.plot(x_grid, [appr(x) for x in x_grid], label='Approximating func')
plt.title(f"s = {s}")
plt.legend()
plt.show()'''
