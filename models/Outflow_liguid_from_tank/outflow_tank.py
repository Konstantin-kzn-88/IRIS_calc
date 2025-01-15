import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def calculate_flow(h0, d_hole_mm, D_tank, P_gauge_initial_MPa, h_gas_initial, rho, t_max, k=1.4, dt=0.1):
    """
    Расчет истечения жидкости с учетом изменения давления газовой подушки
    """
    # Константы
    g = 9.81  # ускорение свободного падения, м/с²
    mu = 0.61  # коэффициент расхода
    A_tank = np.pi * (D_tank / 2) ** 2  # площадь поперечного сечения резервуара, м²
    P_atm = 101325  # атмосферное давление, Па

    # Преобразование диаметра отверстия из мм в м² площади
    A_hole = np.pi * ((d_hole_mm / 1000) / 2) ** 2

    # Преобразование давления из МПа в Па
    P_gauge_initial = P_gauge_initial_MPa * 1e6

    # Массивы для хранения результатов
    t = np.arange(0, t_max, dt)
    h = np.zeros_like(t)  # высота жидкости
    h_gas = np.zeros_like(t)  # высота газовой подушки
    P_gauge = np.zeros_like(t)  # избыточное давление
    Q_mass = np.zeros_like(t)  # массовый расход
    m_flowed = np.zeros_like(t)

    # Начальные условия
    h[0] = h0
    h_gas[0] = h_gas_initial
    P_gauge[0] = P_gauge_initial
    V0_gas = A_tank * h_gas_initial  # начальный объем газа
    V0 = A_tank * h0  # начальный объем жидкости
    m0 = rho * V0  # начальная масса жидкости

    # Время выхода на режим
    t_startup = 0.5

    # Расчет параметров для каждого момента времени
    for i in range(1, len(t)):
        if h[i - 1] <= 0:
            h[i] = 0
            h_gas[i] = h_gas[i - 1]
            P_gauge[i] = P_gauge[i - 1]
            Q_mass[i] = 0
            m_flowed[i] = m_flowed[i - 1]
            continue

        # Расчет давления газовой подушки (адиабатический процесс)
        V_gas = A_tank * (h_gas[i - 1] + h0 - h[i - 1])  # текущий объем газа
        P_gauge[i] = P_gauge_initial * (V0_gas / V_gas) ** k

        # Расчет скорости истечения с учетом давления
        v = np.sqrt(2 * ((P_gauge[i] + rho * g * h[i - 1]) / rho))

        # Расчет массового расхода
        Q_max = mu * A_hole * v * rho

        # Плавный выход на режим
        if t[i] <= t_startup:
            Q_mass[i] = Q_max * (1 - np.exp(-5 * t[i] / t_startup))
        else:
            Q_mass[i] = Q_max

        # Изменение высоты
        dh = (Q_mass[i] * dt) / (rho * A_tank)
        h[i] = max(0, h[i - 1] - dh)

        # Обновление высоты газовой подушки
        h_gas[i] = h_gas_initial + h0 - h[i]

        # Расчет массы вытекшей жидкости
        m_flowed[i] = m_flowed[i - 1] + Q_mass[i] * dt

    return t, h, P_gauge, Q_mass, m_flowed


def plot_results(h0, d_hole_mm, D_tank, P_gauge_initial_MPa, h_gas_initial, rho, t_max):
    # Настройка стиля
    plt.style.use('default')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    # Цветовая схема
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # Расчет параметров
    t, h, P_gauge, Q_mass, m_flowed = calculate_flow(
        h0, d_hole_mm, D_tank, P_gauge_initial_MPa, h_gas_initial, rho, t_max
    )

    # Создание фигуры с сеткой подграфиков
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    fig.suptitle(
        'Истечение жидкости через отверстие\n' +
        f'h₀ = {h0} м, d_отв = {d_hole_mm} мм, D_рез = {D_tank} м\n' +
        f'P_изб.нач = {P_gauge_initial_MPa:.2f} МПа, h_газа = {h_gas_initial} м, ρ = {rho} кг/м³\n' +
        f'Время моделирования = {t_max} с',
        fontsize=14
    )

    # График высоты
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, h, color=colors[0], label='Высота', linewidth=2.5)
    ax1.set_xlabel('Время, с', fontsize=10)
    ax1.set_ylabel('Высота жидкости, м', fontsize=10)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.set_title('Изменение высоты жидкости')

    # График давления
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, P_gauge / 1e6, color=colors[1], label='Давление', linewidth=2.5)
    ax2.set_xlabel('Время, с', fontsize=10)
    ax2.set_ylabel('Избыточное давление, МПа', fontsize=10)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.set_title('Изменение давления газовой подушки')

    # График массового расхода
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, Q_mass, color=colors[2], label='Расход', linewidth=2.5)
    ax3.set_xlabel('Время, с', fontsize=10)
    ax3.set_ylabel('Массовый расход, кг/с', fontsize=10)
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax3.set_title('Изменение массового расхода')

    # График массы вытекшей жидкости
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, m_flowed, color=colors[3], label='Масса вытекшей', linewidth=2.5)
    ax4.set_xlabel('Время, с', fontsize=10)
    ax4.set_ylabel('Масса, кг', fontsize=10)
    ax4.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax4.set_title('Масса вытекшей жидкости')

    # Настройка отступов между графиками
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Параметры задачи
    h0 = 10.0  # начальная высота жидкости, м
    d_hole_mm = 112.8  # диаметр отверстия разгерметизации, мм
    D_tank = 2.0  # диаметр резервуара, м
    P_gauge_initial_MPa = 0.2  # начальное избыточное давление, МПа
    h_gas_initial = 1.0  # начальная высота газовой подушки, м
    rho = 1000  # плотность жидкости, кг/м³
    t_max = 30.0  # время моделирования, с

    # Построение графиков
    fig = plot_results(h0, d_hole_mm, D_tank, P_gauge_initial_MPa, h_gas_initial, rho, t_max)

    # Сохранение графика в файл
    plt.savefig('liquid_flow_results.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Расчет параметров для вывода общей массы
    _, _, _, _, m_flowed = calculate_flow(h0, d_hole_mm, D_tank, P_gauge_initial_MPa, h_gas_initial, rho, t_max)
    print(f"Общая масса вытекшей жидкости за {t_max} с: {m_flowed[-1]:.2f} кг")