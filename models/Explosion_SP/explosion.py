import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Справочные данные
PRESSURES_OF_INTEREST = [100, 70, 28, 14, 5, 2]  # кПа
Z_COEFFICIENT = 0.1  # коэффициент участия горючих газов и паров в горении
Q_SUBSTANCES = {
    'метан': 50240,  # кДж/кг
    'пропан': 46360,  # кДж/кг
    'водород': 120910,  # кДж/кг
    'ацетилен': 48225  # кДж/кг
}


def calculate_probit(delta_P: float, impulse: float) -> float:
    """
    Расчет пробит-функции для избыточного давления и импульса

    Параметры:
    delta_P: избыточное давление (кПа)
    impulse: импульс волны давления (Па·с)

    Возвращает:
    float: значение пробит-функции
    """
    # Переводим кПа в Па
    delta_P_pa = delta_P * 1000
    V = (17500 / delta_P_pa) ** 8.4 + (290 / impulse) ** 9.3
    return 5 - 0.26 * math.log(V)


def calculate_probability(probit: float) -> float:
    """
    Расчет вероятности поражения по значению пробит-функции

    Параметры:
    probit: значение пробит-функции

    Возвращает:
    float: вероятность поражения (от 0 до 1)
    """
    # Используем аппроксимацию нормального распределения
    x = (probit - 5) / math.sqrt(2)
    # Аппроксимация функции ошибок
    z = abs(x)
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - (((((1.061405429 * t + -1.453152027) * t + 1.421413741)
                 * t + -0.284496736) * t + 0.254829592) * t) * math.exp(-z * z)
    probability = 0.5 * (1 + (1 if x > 0 else -1) * erf)
    return probability


def calculate_equivalent_mass(m: float, Q_combustion: float) -> float:
    """
    Расчет приведенной массы газа или пара

    Параметры:
    m: масса горючих газов/паров, поступивших в результате аварии (кг)
    Q_combustion: удельная теплота сгорания газа или пара (кДж/кг)

    Возвращает:
    float: приведенная масса газа или пара (кг)
    """
    Q0 = 4520  # константа, кДж/кг (4.52e6 / 1000)
    return (Q_combustion / Q0) * m * Z_COEFFICIENT


def calculate_excess_pressure(m_pr: float, r: float, P0: float = 101.0) -> float:
    """
    Расчет избыточного давления при сгорании газопаровоздушных смесей

    Параметры:
    m_pr: приведенная масса газа или пара (кг)
    r: расстояние от геометрического центра газопаровоздушного облака (м)
    P0: атмосферное давление (кПа), по умолчанию 101 кПа

    Возвращает:
    float: избыточное давление (кПа)
    """
    MAX_PRESSURE = 700  # максимальное избыточное давление (кПа)

    if r < 0.1:  # предотвращаем деление на очень маленькие значения
        return MAX_PRESSURE

    term1 = 0.8 * (m_pr ** 0.33) / r
    term2 = 3 * (m_pr ** 0.66) / (r ** 2)
    term3 = 5 * m_pr / (r ** 3)

    pressure = P0 * (term1 + term2 + term3)
    return min(pressure, MAX_PRESSURE)  # ограничиваем максимальное давление


def calculate_pressure_impulse(m_pr: float, r: float) -> float:
    """
    Расчет импульса волны давления

    Параметры:
    m_pr: приведенная масса газа или пара (кг)
    r: расстояние от геометрического центра газопаровоздушного облака (м)

    Возвращает:
    float: импульс волны давления (Па·с)
    """
    return 123 * (m_pr ** 0.66) / r


def find_distance_for_pressure(m_pr: float, target_pressure: float, P0: float = 101.0) -> float:
    """
    Поиск расстояния для заданного давления методом бинарного поиска

    Параметры:
    m_pr: приведенная масса газа или пара (кг)
    target_pressure: целевое давление (кПа)
    P0: атмосферное давление (кПа)

    Возвращает:
    float: расстояние (м)
    """
    left, right = 0.1, 1000.0  # начальный диапазон поиска
    tolerance = 0.1  # допустимая погрешность

    while right - left > tolerance:
        mid = (left + right) / 2
        pressure = calculate_excess_pressure(m_pr, mid, P0)

        if abs(pressure - target_pressure) < tolerance:
            return mid
        elif pressure > target_pressure:
            left = mid
        else:
            right = mid

    return (left + right) / 2


def calculate_damage_probability(delta_P: float, impulse: float) -> Tuple[float, float]:
    """
    Расчет вероятности поражения через пробит-функцию

    Параметры:
    delta_P: избыточное давление (кПа)
    impulse: импульс волны давления (Па·с)

    Возвращает:
    Tuple[float, float]: (значение пробит-функции, вероятность поражения)
    """
    probit = calculate_probit(delta_P, impulse)
    probability = calculate_probability(probit)
    return probit, probability


def calculate_distances_for_pressures(mass: float, Q_combustion: float,
                                      pressures: List[float] = PRESSURES_OF_INTEREST) -> Dict[float, dict]:
    """
    Расчет расстояний для заданных значений давления

    Параметры:
    mass: масса горючих газов/паров (кг)
    Q_combustion: удельная теплота сгорания (кДж/кг)
    pressures: список давлений (кПа)

    Возвращает:
    dict: словарь {давление: {расстояние, импульс, пробит, вероятность}}
    """
    m_pr = calculate_equivalent_mass(mass, Q_combustion)
    results = {}

    for p in pressures:
        dist = find_distance_for_pressure(m_pr, p)
        impulse = calculate_pressure_impulse(m_pr, dist)
        probit, probability = calculate_damage_probability(p, impulse)

        results[p] = {
            'distance': dist,
            'impulse': impulse,
            'probit': probit,
            'probability': probability
        }

    return results


def calculate_all_parameters(mass: float, substance: str, distance: float) -> dict:
    """
    Расчет всех параметров взрыва

    Параметры:
    mass: масса горючих газов/паров (кг)
    substance: название вещества из справочника Q_SUBSTANCES
    distance: расстояние от центра облака (м)

    Возвращает:
    dict: словарь с результатами расчетов
    """
    if substance not in Q_SUBSTANCES:
        raise ValueError(f"Вещество {substance} отсутствует в справочнике")

    Q_combustion = Q_SUBSTANCES[substance]
    m_pr = calculate_equivalent_mass(mass, Q_combustion)
    excess_pressure = calculate_excess_pressure(m_pr, distance)
    pressure_impulse = calculate_pressure_impulse(m_pr, distance)
    probit, probability = calculate_damage_probability(excess_pressure, pressure_impulse)

    distances = calculate_distances_for_pressures(mass, Q_combustion)

    return {
        "equivalent_mass": m_pr,
        "excess_pressure": excess_pressure,
        "pressure_impulse": pressure_impulse,
        "probit": probit,
        "probability": probability,
        "distances_for_pressures": distances
    }


def plot_blast_parameters(mass: float, substance: str, output_dir: str = "./"):
    """
    Построение графиков параметров взрыва

    Параметры:
    mass: масса горючих газов/паров (кг)
    substance: название вещества из справочника Q_SUBSTANCES
    output_dir: директория для сохранения графиков
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    # Создаем массив расстояний с логарифмическим масштабом для детального изучения малых расстояний
    distances = np.logspace(np.log10(0.1), np.log10(200), 300)

    # Получаем данные для графиков
    if substance not in Q_SUBSTANCES:
        raise ValueError(f"Вещество {substance} отсутствует в справочнике")

    Q_combustion = Q_SUBSTANCES[substance]
    m_pr = calculate_equivalent_mass(mass, Q_combustion)

    # Рассчитываем параметры для каждого расстояния
    pressures = [calculate_excess_pressure(m_pr, r) for r in distances]
    impulses = [calculate_pressure_impulse(m_pr, r) for r in distances]
    probits = [calculate_probit(p, i) for p, i in zip(pressures, impulses)]
    probabilities = [calculate_probability(pr) * 100 for pr in probits]  # в процентах

    # Создаем фигуру с 6 графиками (добавляем логарифмические графики для давления)
    fig = plt.figure(figsize=(20, 15))
    grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.3)

    # График избыточного давления (линейный масштаб)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(distances, pressures, 'b-', linewidth=2, label='Давление')
    ax1.axhline(y=700, color='r', linestyle='--', label='Макс. давление (700 кПа)')
    ax1.set_xlabel('Расстояние (м)')
    ax1.set_ylabel('Избыточное давление (кПа)')
    ax1.set_title('Зависимость давления от расстояния\n(линейный масштаб)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend()

    # График избыточного давления (логарифмический масштаб)
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.loglog(distances, pressures, 'b-', linewidth=2, label='Давление')
    ax2.axhline(y=700, color='r', linestyle='--', label='Макс. давление (700 кПа)')
    ax2.set_xlabel('Расстояние (м)')
    ax2.set_ylabel('Избыточное давление (кПа)')
    ax2.set_title('Зависимость давления от расстояния\n(логарифмический масштаб)')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend()

    # График импульса
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.semilogx(distances, impulses, 'g-', linewidth=2)
    ax3.set_xlabel('Расстояние (м)')
    ax3.set_ylabel('Импульс (Па·с)')
    ax3.set_title('Зависимость импульса от расстояния')
    ax3.grid(True, which='both', linestyle='--', alpha=0.7)

    # График пробит-функции
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.semilogx(distances, probits, 'r-', linewidth=2)
    ax4.set_xlabel('Расстояние (м)')
    ax4.set_ylabel('Значение пробит-функции')
    ax4.set_title('Зависимость пробит-функции от расстояния')
    ax4.grid(True, which='both', linestyle='--', alpha=0.7)

    # График вероятности поражения
    ax5 = fig.add_subplot(grid[2, 0])
    ax5.semilogx(distances, probabilities, 'y-', linewidth=2)
    ax5.set_xlabel('Расстояние (м)')
    ax5.set_ylabel('Вероятность поражения (%)')
    ax5.set_title('Зависимость вероятности поражения от расстояния')
    ax5.grid(True, which='both', linestyle='--', alpha=0.7)
    ax5.set_ylim([0, 100])

    # Добавляем характерные значения давления
    ax6 = fig.add_subplot(grid[2, 1])
    characteristic_distances = []
    characteristic_pressures = [100, 70, 28, 14, 5, 2]

    for p in characteristic_pressures:
        r = find_distance_for_pressure(m_pr, p)
        characteristic_distances.append(r)

    ax6.plot(characteristic_distances, characteristic_pressures, 'ko-', label='Характерные точки')
    ax6.set_xlabel('Расстояние (м)')
    ax6.set_ylabel('Давление (кПа)')
    ax6.set_title('Характерные значения давления')
    ax6.grid(True)
    for i, p in enumerate(characteristic_pressures):
        ax6.annotate(f'{p} кПа',
                     (characteristic_distances[i], characteristic_pressures[i]),
                     xytext=(5, 5), textcoords='offset points')

    # fig.suptitle(f'Параметры взрыва {substance} массой {mass} кг', fontsize=16, y=0.95)

    # Сохраняем графики
    output_path = f"{output_dir}/blast_parameters_{mass}kg.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


# Пример использования:
if __name__ == "__main__":
    # Пример расчета для метана
    mass = 100  # кг
    distance = 20  # метров
    substance = 'метан'

    # Расчет параметров
    results = calculate_all_parameters(mass, substance, distance)

    print(f"\nРезультаты расчета для {substance} массой {mass} кг:")
    print(f"Коэффициент участия: {Z_COEFFICIENT}")
    print(f"Приведенная масса: {results['equivalent_mass']:.2f} кг")
    print(f"Избыточное давление на расстоянии {distance}м: {results['excess_pressure']:.2f} кПа")
    print(f"Импульс волны давления: {results['pressure_impulse']:.2f} Па·с")
    print(f"Пробит-функция: {results['probit']:.2f}")
    print(f"Вероятность поражения: {results['probability']:.2%}")

    print("\nРасстояния для заданных давлений:")
    for pressure, data in results['distances_for_pressures'].items():
        print(f"\nДавление {pressure} кПа:")
        print(f"  Расстояние: {data['distance']:.2f} м")
        print(f"  Импульс: {data['impulse']:.2f} Па·с")
        print(f"  Пробит-функция: {data['probit']:.2f}")
        print(f"  Вероятность поражения: {data['probability']:.2%}")

    # Построение и сохранение графиков
    output_path = plot_blast_parameters(mass, substance)
    print(f"\nГрафики сохранены в файл: {output_path}")