import numpy as np
import matplotlib
matplotlib.use('Agg')  # Добавляем эту строку перед импортом pyplot
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


def validate_inputs(**params):
    """
    Проверка корректности входных параметров

    Args:
        **params: Словарь параметров для проверки

    Raises:
        ValueError: Если какой-либо параметр имеет недопустимое значение
    """
    for name, value in params.items():
        if value <= 0:
            raise ValueError(f"Параметр {name} должен быть положительным")
        if name == 'theta' and (value < 0 or value > np.pi / 2):
            raise ValueError(f"Угол наклона должен быть между 0 и 90 градусов")
        if name.startswith('P') and value < 0:
            raise ValueError(f"Давление не может быть отрицательным")


def calculate_reynolds(Q: float, d: float, rho: float, mu: float) -> float:
    """
    Расчет числа Рейнольдса

    Args:
        Q: Расход жидкости (м³/с)
        d: Диаметр трубы (м)
        rho: Плотность жидкости (кг/м³)
        mu: Динамическая вязкость (Па·с)

    Returns:
        float: Число Рейнольдса
    """
    velocity = Q / (np.pi * d ** 2 / 4)
    return rho * velocity * d / mu


def plot_results(pressures: np.ndarray, x: np.ndarray, Q: np.ndarray,
                 d: np.ndarray, params: Dict, fig=None) -> Optional[plt.Figure]:
    """
    Создание информативных графиков

    Args:
        pressures: Массив значений давления
        x: Массив координат по длине скважины
        Q: Массив значений расхода
        d: Массив значений диаметра
        params: Словарь с параметрами расчета
        fig: объект Figure для построения графиков (если None, создается новый)

    Returns:
        Optional[plt.Figure]: объект Figure с графиками
    """
    # Преобразуем координату x в глубину
    depth = params['L'] - x  # теперь 0 наверху (на устье), а L внизу (на забое)

    if fig is None:
        fig = plt.figure(figsize=(15, 12))

    axes = fig.subplots(2, 2)

    # График давления
    axes[0, 0].plot(pressures / 1e6, depth, 'b-', linewidth=2)
    axes[0, 0].grid(True)
    axes[0, 0].set_xlabel("Давление, МПа")
    axes[0, 0].set_ylabel("Глубина, м")
    axes[0, 0].set_title("Распределение давления")

    # График расхода
    axes[0, 1].plot(Q * 1000, depth, 'r-', linewidth=2)  # перевод в л/с
    axes[0, 1].grid(True)
    axes[0, 1].set_xlabel("Расход, л/с")
    axes[0, 1].set_ylabel("Глубина, м")
    axes[0, 1].set_title("Распределение расхода")

    # График скорости
    velocity = Q / (np.pi * d ** 2 / 4)
    axes[1, 0].plot(velocity, depth, 'g-', linewidth=2)
    axes[1, 0].grid(True)
    axes[1, 0].set_xlabel("Скорость, м/с")
    axes[1, 0].set_ylabel("Глубина, м")
    axes[1, 0].set_title("Распределение скорости")

    # График числа Рейнольдса
    Re = calculate_reynolds(Q, d, params['rho'], params.get('mu', 0.001))
    axes[1, 1].plot(Re, depth, 'm-', linewidth=2)
    axes[1, 1].grid(True)
    axes[1, 1].set_xlabel("Число Рейнольдса")
    axes[1, 1].set_ylabel("Глубина, м")
    axes[1, 1].set_title("Распределение числа Рейнольдса")

    # Добавление информации о параметрах
    info_text = (
        f"Параметры расчета:\n"
        f"L = {params['L']} м\n"
        f"d0 = {params['d0']} м\n"
        f"d1 = {params['d1']} м\n"
        f"θ = {np.rad2deg(params['theta']):.1f}°\n"
        f"P0 = {params['P0'] / 1e6:.1f} МПа\n"
        f"ρ = {params['rho']} кг/м³\n"
        f"μ = {params.get('mu', 0.001)} Па·с"
    )
    fig.text(0.02, 0.02, info_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def calc_flow_rate_and_pressure(P0: float, P_atm: float, L: float, d0: float,
                                d1: float, rho: float, g: float, theta: float,
                                mu: float = 0.001) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчет расхода и распределения давления в наклонной скважине.

    Args:
        P0: давление в пласте (Па)
        P_atm: атмосферное давление (Па)
        L: длина скважины (м)
        d0: диаметр на устье (м)
        d1: диаметр на забое (м)
        rho: плотность жидкости (кг/м³)
        g: ускорение свободного падения (м/с²)
        theta: угол наклона (рад)
        mu: динамическая вязкость (Па·с)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: расход, давление и координаты
    """
    try:
        validate_inputs(L=L, d0=d0, d1=d1, rho=rho, P0=P0, P_atm=P_atm, theta=theta)

        # Создание сетки
        n_points = 100
        x = np.linspace(0, L, n_points)
        d = d0 + (d1 - d0) * x / L

        # Учет гидростатического давления
        P_hydrostatic = rho * g * np.sin(theta) * (L - x)

        # Распределение давления с учетом граничных условий
        P = P0 * (1 - x / L) + P_atm * (x / L) + P_hydrostatic

        # Расчет расхода с учетом вязкости
        dP = P[:-1] - P[1:]  # перепад давления между точками
        dx = x[1] - x[0]  # расстояние между точками
        d_avg = (d[:-1] + d[1:]) / 2  # средний диаметр

        # Формула Пуазейля для расхода с учетом вязкости
        Q = np.pi * d_avg ** 4 * dP / (128 * mu * dx)

        # Добавляем значение расхода для последней точки
        Q = np.append(Q, Q[-1])

        return Q, P, x

    except Exception as e:
        print(f"Ошибка при расчете: {str(e)}")
        return np.zeros(n_points), np.zeros(n_points), x


def save_results(x: np.ndarray, P: np.ndarray, Q: np.ndarray,
                 filename: str = 'results.txt') -> None:
    """
    Сохранение результатов расчета в файл

    Args:
        x: координаты по длине скважины
        P: давление
        Q: расход
        filename: имя файла для сохранения
    """
    header = "x(m)\tP(Pa)\tQ(m3/s)"
    data = np.column_stack((x, P, Q))
    np.savetxt(filename, data, header=header, delimiter='\t', encoding='ascii')


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
    'mu': 0.001  # динамическая вязкость, Па·с
}

# Расчет
Q, P, x = calc_flow_rate_and_pressure(**params)
print(f"Интенсивность истечения: {Q[-1]:.2f} м³/с")

# Визуализация результатов
d = params['d0'] + (params['d1'] - params['d0']) * x / params['L']
plot_results(P, x, Q, d, params)

# Сохранение результатов
save_results(x, P, Q)

# plt.show()