import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def friction_factor(Re: float, roughness: float, d: float) -> float:
    """
    Расчет коэффициента трения с учетом режима течения
    """
    if Re <= 0:  # Защита от некорректных значений
        return 0.0

    if Re < 2300:  # Ламинарный режим
        return 64 / Re
    elif Re > 4000:  # Турбулентный режим
        relative_roughness = roughness / d
        # Начальное приближение по формуле Никурадзе
        f = (1.8 * np.log10(Re) - 1.5) ** (-2)

        # Уточнение по формуле Колбрука-Уайта
        for _ in range(10):
            f_new = (-2 * np.log10(relative_roughness / 3.7 + 2.51 / (Re * np.sqrt(f)))) ** (-2)
            if abs(f - f_new) < 1e-6:
                return f_new
            f = f_new
        return f
    else:  # Переходный режим
        # Плавная интерполяция между режимами
        f_lam = 64 / 2300
        Re_t = 4000
        f_turb = (-2 * np.log10(roughness / (3.7 * d) + 2.51 / (Re_t * np.sqrt(0.032)))) ** (-2)
        w = (Re - 2300) / (Re_t - 2300)
        return f_lam * (1 - w) + f_turb * w


def calc_flow_rate_and_pressure(P0: float, P_atm: float, L: float, d0: float,
                                d1: float, rho: float, g: float, theta: float,
                                mu: float = 0.001, roughness: float = 1e-5,
                                n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчет расхода и распределения давления в наклонной скважине
    с улучшенным алгоритмом
    """
    # Создаем сетку и массивы для расчета
    x = np.linspace(0, L, n_points)
    dx = x[1] - x[0]
    d = d0 + (d1 - d0) * x / L  # Линейное изменение диаметра
    A = np.pi * d ** 2 / 4

    # Инициализация массивов
    P = np.zeros(n_points)
    Q = np.zeros(n_points)
    P[0] = P0  # Давление на забое

    # Предварительная оценка расхода
    # Используем уравнение Бернулли для грубой оценки
    P_hydrostatic = rho * g * np.sin(theta) * L
    dP_available = P0 - P_atm - P_hydrostatic
    if dP_available <= 0:
        dP_available = P0 - P_atm

    # Начальное приближение расхода через минимальное сечение
    A_min = np.pi * min(d0, d1) ** 2 / 4
    v_max = np.sqrt(2 * abs(dP_available) / rho)
    Q_init = 0.6 * A_min * v_max  # Коэффициент 0.6 для учета потерь
    Q.fill(Q_init)

    # Основной итерационный процесс
    max_iter = 50
    convergence = False
    relaxation = 0.3  # Коэффициент релаксации

    for iteration in range(max_iter):
        Q_old = Q.copy()
        P[0] = P0

        # Прямой проход - расчет давления
        for i in range(n_points - 1):
            # Расчет локальных параметров
            A_avg = (A[i] + A[i + 1]) / 2
            v = Q[i] / A[i]
            Re = abs(rho * v * d[i] / mu) if v != 0 else 0

            # Потери на трение
            f = friction_factor(Re, roughness, d[i])
            dP_friction = f * rho * v * abs(v) * dx / (2 * d[i])

            # Потери на местные сопротивления (сужение/расширение)
            if i < n_points - 1:
                if d[i + 1] < d[i]:  # сужение
                    xi = 0.5 * (1 - (d[i + 1] / d[i]) ** 2)
                else:  # расширение
                    xi = (1 - (d[i] / d[i + 1]) ** 2)
                dP_local = xi * rho * v * abs(v) / 2
            else:
                dP_local = 0

            # Гидростатическая составляющая
            dP_hydrostatic = rho * g * np.sin(theta) * dx

            # Суммарное падение давления
            dP_total = dP_friction + dP_local + dP_hydrostatic

            # Обновление давления
            P[i + 1] = P[i] - dP_total

            # Расчет нового расхода
            if i < n_points - 1:
                dP = P[i] - P[i + 1]
                # Используем формулу с учетом потерь
                Q_new = A_avg * np.sqrt(2 * abs(dP) / (rho * (1 + f * dx / d[i] + xi)))
                # Применяем релаксацию для стабильности
                Q[i + 1] = Q[i] * (1 - relaxation) + Q_new * relaxation

        # Проверка сходимости
        rel_error = np.max(np.abs(Q - Q_old) / (Q + 1e-10))
        if rel_error < 1e-4:
            convergence = True
            break

    # Финальная корректировка давления (не должно быть ниже атмосферного)
    P = np.maximum(P, P_atm)

    return Q, P, x


def plot_results(x: np.ndarray, P: np.ndarray, Q: np.ndarray, d: np.ndarray,
                 velocity: np.ndarray, Re: np.ndarray, params: dict) -> None:
    """
    Визуализация результатов расчета
    """
    # Преобразуем координату x в глубину
    depth = params['L'] - x  # теперь 0 наверху, а L внизу

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.set_facecolor('white')

    # График давления
    axes[0, 0].plot(P / 1e6, depth, 'b-', linewidth=2)
    axes[0, 0].grid(True)
    axes[0, 0].set_xlabel("Давление, МПа")
    axes[0, 0].set_ylabel("Глубина, м")
    axes[0, 0].set_title("Распределение давления")
    axes[0, 0].invert_yaxis()

    # График расхода
    axes[0, 1].plot(Q * 1000, depth, 'r-', linewidth=2)  # перевод в л/с
    axes[0, 1].grid(True)
    axes[0, 1].set_xlabel("Расход, л/с")
    axes[0, 1].set_ylabel("Глубина, м")
    axes[0, 1].set_title("Распределение расхода")
    axes[0, 1].invert_yaxis()

    # График дебита
    axes[1, 0].plot(Q, depth, 'g-', linewidth=2)
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel("Дебит, м³/с")
    axes[1, 0].set_ylabel("Глубина, м")
    axes[1, 0].set_title("Распределение дебита")
    axes[1, 0].invert_yaxis()

    # График числа Рейнольдса
    axes[1, 1].plot(Re / 1e3, depth, 'm-', linewidth=2)  # перевод в тысячи
    axes[1, 1].grid(True)
    axes[1, 1].set_xlabel("Число Рейнольдса, ×10³")
    axes[1, 1].set_ylabel("Глубина, м")
    axes[1, 1].set_title("Распределение числа Рейнольдса")
    axes[1, 1].invert_yaxis()

    # Добавление информации о параметрах
    info_text = (
        f"Параметры расчета:\n"
        f"L = {params['L']} м\n"
        f"d0 = {params['d0']} м\n"
        f"d1 = {params['d1']} м\n"
        f"θ = {np.rad2deg(params['theta']):.1f}°\n"
        f"P0 = {params['P0'] / 1e6:.1f} МПа\n"
        f"ρ = {params['rho']} кг/м³\n"
        f"μ = {params['mu']} Па·с\n"
        f"k = {params['roughness'] * 1000:.2f} мм"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


# Физические параметры
params = {
    'rho': 1000,  # плотность жидкости, кг/м³
    'g': 9.81,  # ускорение свободного падения, м/с²
    'theta': np.deg2rad(30),  # наклон скважины, радианы
    'L': 1000,  # длина скважины, м
    'd0': 0.2,  # диаметр на устье, м
    'd1': 0.1,  # диаметр на забое, м
    'P0': 10e6,  # давление в пласте, Па
    'P_atm': 101325,  # атмосферное давление, Па
    'mu': 0.001,  # динамическая вязкость, Па·с
    'roughness': 1e-5  # шероховатость стенок, м
}

# Выполняем расчет
Q, P, x = calc_flow_rate_and_pressure(**params)
d = params['d0'] + (params['d1'] - params['d0']) * x / params['L']
velocity = Q / (np.pi * d ** 2 / 4)
Re = params['rho'] * velocity * d / params['mu']

# print(f"\nРезультаты расчета:")
# print(f"Расход на устье: {Q[-1]:.3f} м³/с ({Q[-1] * 1000:.1f} л/с)")
# print(f"Скорость на устье: {velocity[-1]:.2f} м/с")
# print(f"Число Рейнольдса на устье: {Re[-1]:.0f}")
# print(f"Давление на забое: {P[0] / 1e6:.2f} МПа")  # индекс 0 - забой
# print(f"Давление на устье: {P[-1] / 1e6:.2f} МПа")  # индекс -1 - устье

# Создаем и сохраняем графики
fig = plot_results(x, P, Q, d, velocity, Re, params)
plt.savefig('well_flow_results.png', dpi=300, bbox_inches='tight')
plt.close()