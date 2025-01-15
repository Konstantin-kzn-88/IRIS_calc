# -----------------------------------------------------------
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда Agg перед импортом pyplot
# Класс предназначен для расчета пожара пролива на открытой местности
#
# Приказ МЧС № 404 от 10.07.2009
# (C) 2022 Kuznetsov Konstantin, Kazan, Russian Federation
# email kuznetsovkm@yandex.ru
# -----------------------------------------------------------

import math
from typing import Tuple, List, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt


def get_nearest_value(array: List[float], value: float) -> float:
    """
    Поиск ближайшего значения в массиве
    """
    return min(array, key=lambda x: abs(x - value))


class Probit:
    """
    Класс для расчета пробит-функции и вероятности поражения
    """

    def probability(self, probit: float) -> float:
        """Расчет вероятности поражения"""
        return 0.5 * (1.0 + math.erf((probit - 5.0) / math.sqrt(2.0)))

    def probit_strait_fire(self, dist: float, q: float, t_exp: float = 30) -> float:
        """
        Расчет пробит-функции для теплового излучения

        Args:
            dist: расстояние до точки с заданной интенсивностью, м
            q: интенсивность теплового излучения, кВт/м²
            t_exp: время экспозиции, с (по умолчанию 30с)
        """
        return -12.8 + 2.56 * math.log(q ** 1.33 * t_exp)


@dataclass
class InputParameters:
    """
    Класс для хранения и валидации входных параметров расчета
    """
    S_spill: float  # площадь пролива, м2
    m_sg: float  # удельная плотность выгорания, кг/(с*м2)
    mol_mass: float  # молекулярная масса, кг/кмоль
    t_boiling: float  # температура кипения, град.С
    wind_velocity: float  # скорость ветра, м/с

    def __post_init__(self):
        """Валидация входных параметров после инициализации"""
        self._validate_parameters()

    def _validate_parameters(self):
        """Проверка корректности входных параметров"""
        if any(param <= 0 for param in [self.S_spill, self.m_sg, self.mol_mass, self.wind_velocity]):
            raise ValueError('Все параметры должны быть положительными числами')

        if self.t_boiling < -273.15:  # Абсолютный нуль
            raise ValueError('Температура кипения не может быть ниже абсолютного нуля')

        if self.mol_mass > 1000:  # Примерное ограничение для реальных веществ
            raise ValueError('Слишком большая молекулярная масса')


class StraitFire:
    """
    Класс для расчета зон действия пожара пролива в зависимости от расстояния
    до облучаемого объекта.

    Методика расчета основана на Приказе МЧС № 404 от 10.07.2009.
    """

    # Константы класса
    MINIMAL_SAFE_DISTANCE = 0.1  # минимальное безопасное расстояние, м
    MINIMAL_RADIATION = 1.2  # минимальная учитываемая интенсивность излучения, кВт/м2
    CLASSIFIED_ZONES = [10.5, 7.0, 4.2, 1.4]  # границы классифицированных зон, кВт/м2

    def __init__(self):
        """Инициализация класса"""
        self.results_cache = {}  # кэш для хранения результатов расчетов
        self.probit_calculator = Probit()

    def calculate_effective_diameter(self, S_spill: float) -> float:
        """Расчет эффективного диаметра пролива"""
        return math.sqrt(4 * S_spill / math.pi)

    def termal_radiation_point(self, params: InputParameters, radius: float) -> float:
        """
        Расчет интенсивности теплового излучения в заданной точке

        Args:
            params: параметры расчета
            radius: расстояние от геометрического центра пролива, м

        Returns:
            float: интенсивность теплового излучения, кВт/м2
        """
        if radius < 0:
            raise ValueError('Радиус не может быть отрицательным')

        # Расчет эффективного диаметра
        D_eff = self.calculate_effective_diameter(params.S_spill)

        # Проверка не попадает ли точка в пролив
        if radius < (D_eff / 2 + self.MINIMAL_SAFE_DISTANCE):
            radius = D_eff / 2 + self.MINIMAL_SAFE_DISTANCE

        # Расчет плотности паров
        po_steam = params.mol_mass / (22.413 * (1 + 0.00367 * params.t_boiling))

        # Расчет безразмерной скорости ветра
        u_star = params.wind_velocity / math.pow(
            (params.m_sg * 9.8 * D_eff) / po_steam,
            1 / 3
        )

        # Расчет длины пламени
        flame_length = self._calculate_flame_length(D_eff, params.m_sg, u_star)

        # Расчет угла наклона пламени
        cos_tetta = min(1, math.pow(u_star, -0.5) if u_star >= 1 else 1)
        tetta = math.acos(cos_tetta)

        # Расчет промежуточных параметров
        a_pr = 2 * flame_length / D_eff
        b_pr = 2 * radius / D_eff

        # Расчет угловых коэффициентов
        Fv, Fh = self._calculate_angular_coefficients(a_pr, b_pr, tetta)
        Fq = math.sqrt(Fv ** 2 + Fh ** 2)

        # Расчет коэффициента пропускания атмосферы
        tay = math.exp(-7e-4 * (radius - 0.5 * D_eff))

        # Интенсивность теплового излучения
        E_f = 25  # среднеповерхностная плотность теплового излучения
        q_term = Fq * tay * E_f

        return round(q_term, 2)

    def _calculate_flame_length(self, D_eff: float, m_sg: float, u_star: float) -> float:
        """Расчет длины пламени"""
        base_coefficient = m_sg / (1.15 * math.sqrt(9.81 * D_eff))

        if u_star >= 1:
            return 55 * D_eff * math.pow(base_coefficient, 0.67) * math.pow(u_star, 0.21)
        else:
            return 42 * D_eff * math.pow(base_coefficient, 0.61)

    def _calculate_angular_coefficients(self, a_pr: float, b_pr: float, tetta: float) -> Tuple[float, float]:
        """Расчет угловых коэффициентов облученности"""
        # Расчет промежуточных параметров
        A_pr = math.sqrt(a_pr ** 2 + (b_pr + 1) ** 2 - 2 * a_pr * (b_pr + 1) * math.sin(tetta))
        B_pr = math.sqrt(a_pr ** 2 + (b_pr - 1) ** 2 - 2 * a_pr * (b_pr - 1) * math.sin(tetta))
        C_pr = math.sqrt(1 + (b_pr ** 2 - 1) * math.cos(tetta) ** 2)
        D_pr = math.sqrt((b_pr - 1) / (b_pr + 1))
        E_pr = (a_pr * math.cos(tetta)) / (b_pr - a_pr * math.sin(tetta))
        F_pr = math.sqrt(b_pr ** 2 - 1)

        # Расчет вертикального углового коэффициента
        Fv = self._calculate_vertical_coefficient(E_pr, D_pr, a_pr, b_pr, tetta, A_pr, B_pr, C_pr, F_pr)

        # Расчет горизонтального углового коэффициента
        Fh = self._calculate_horizontal_coefficient(D_pr, tetta, C_pr, a_pr, b_pr, F_pr, A_pr, B_pr)

        return Fv, Fh

    def _calculate_vertical_coefficient(self, E_pr, D_pr, a_pr, b_pr, tetta, A_pr, B_pr, C_pr, F_pr) -> float:
        """Расчет вертикального углового коэффициента облученности"""
        return (1 / math.pi) * (
                -E_pr * math.atan(D_pr) +
                E_pr * ((a_pr ** 2 + (b_pr + 1) ** 2 - 2 * b_pr * (1 + a_pr * math.sin(tetta))) / (A_pr * B_pr)) *
                math.atan((A_pr * D_pr) / B_pr) +
                (math.cos(tetta) / C_pr) * (
                        math.atan((a_pr * b_pr - F_pr ** 2 * math.sin(tetta)) / (F_pr * C_pr)) +
                        math.atan(F_pr ** 2 * math.sin(tetta) / (F_pr * C_pr))
                )
        )

    def _calculate_horizontal_coefficient(self, D_pr, tetta, C_pr, a_pr, b_pr, F_pr, A_pr, B_pr) -> float:
        """Расчет горизонтального углового коэффициента облученности"""
        return (1 / math.pi) * (
                math.atan(1 / D_pr) +
                (math.sin(tetta) / C_pr) * (
                        math.atan((a_pr * b_pr - F_pr ** 2 * math.sin(tetta)) / (F_pr * C_pr)) +
                        math.atan((F_pr ** 2 * math.sin(tetta)) / (F_pr * C_pr))
                ) -
                ((a_pr ** 2 + (b_pr + 1) ** 2 - 2 * (b_pr + 1 + a_pr * b_pr * math.sin(tetta))) / (A_pr * B_pr)) *
                math.atan(A_pr * D_pr / B_pr)
        )

    def termal_radiation_array(self, params: InputParameters) -> Tuple[
        List[float], List[float], List[float], List[float]]:
        """
        Расчет массивов параметров теплового излучения на различных расстояниях

        Returns:
            Tuple[List[float], List[float], List[float], List[float]]:
            (радиусы, интенсивность излучения, пробит-функция, вероятность поражения)
        """
        # Проверяем наличие результата в кэше
        cache_key = (params.S_spill, params.m_sg, params.mol_mass, params.t_boiling, params.wind_velocity)
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]

        radius_arr = []
        q_term_arr = []
        probit_arr = []
        probability_arr = []

        radius = self.MINIMAL_SAFE_DISTANCE
        q_term = self.termal_radiation_point(params, radius)

        while q_term > self.MINIMAL_RADIATION:
            q_term = self.termal_radiation_point(params, radius)
            q_term_arr.append(q_term)
            radius_arr.append(round(radius, 2))
            radius += 0.1

        # Расчет пробит-функции и вероятности поражения
        D_eff = self.calculate_effective_diameter(params.S_spill)
        r_4_kw = next((r for r, q in zip(radius_arr, q_term_arr) if q < 4), 0)

        for i, radius in enumerate(radius_arr):
            if radius < D_eff:
                probit = 8.09
                probability = 0.99
            else:
                dist = r_4_kw - radius
                if dist < 0:
                    probit = probability = 0
                else:
                    probit = self.probit_calculator.probit_strait_fire(dist, q_term_arr[i])
                    probability = self.probit_calculator.probability(probit)

            probit_arr.append(probit)
            probability_arr.append(probability)

        result = (radius_arr, q_term_arr, probit_arr, probability_arr)
        self.results_cache[cache_key] = result
        return result

    def termal_class_zone(self, params: InputParameters) -> List[float]:
        """
        Определение радиусов классифицированных зон поражения

        Returns:
            List[float]: список радиусов для каждой классифицированной зоны
        """
        radius_arr, q_term_arr, _, _ = self.termal_radiation_array(params)
        radius_CZA = []

        for zone_value in self.CLASSIFIED_ZONES:
            ind = q_term_arr.index(get_nearest_value(q_term_arr, zone_value))
            radius_CZA.append(radius_arr[ind])

        return radius_CZA

    def plot_results(self, params: InputParameters):
        """
        Построение графиков зависимостей для презентации
        """
        radius_arr, q_term_arr, probit_arr, probability_arr = self.termal_radiation_array(params)

        # Создаем фигуру с тремя подграфиками
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16))

        # Общий заголовок
        # fig.suptitle('Расчет параметров пожара пролива\nна открытой местности',
        #              fontsize=16, fontweight='bold', y=0.95)

        # Цветовая схема
        colors = ['#2E86C1', '#E74C3C', '#27AE60']  # Синий, Красный, Зеленый

        # Настройка основного стиля
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f9fa')
            for spine in ax.spines.values():
                spine.set_color('#cccccc')

        # График интенсивности теплового излучения
        ax1.plot(radius_arr, q_term_arr, color=colors[0], linewidth=2.5,
                 label='Интенсивность теплового излучения')
        ax1.set_xlabel('Расстояние, м', fontsize=10)
        ax1.set_ylabel('Интенсивность,\nкВт/м²', fontsize=10)
        ax1.set_title('Зависимость интенсивности теплового излучения от расстояния',
                      fontsize=12, pad=20)
        ax1.legend(fontsize=10)

        # Добавляем исходные данные
        info_text = (
            f"Исходные параметры:\n\n"
            f"Площадь пролива: {params.S_spill} м²\n"
            f"Скорость выгорания: {params.m_sg} кг/(с·м²)\n"
            f"Молекулярная масса: {params.mol_mass} кг/кмоль\n"
            f"Температура кипения: {params.t_boiling} °C\n"
            f"Скорость ветра: {params.wind_velocity} м/с"
        )

        # Размещаем текст в левом нижнем углу
        ax1.text(0.02, 0.02, info_text,
                 transform=ax1.transAxes,
                 verticalalignment='bottom',
                 horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.7',
                           facecolor='white',
                           edgecolor='gray',
                           alpha=0.8),
                 fontsize=10)

        # График вероятности поражения
        ax2.plot(radius_arr, probability_arr, color=colors[1], linewidth=2.5,
                 label='Вероятность поражения')
        ax2.set_xlabel('Расстояние, м', fontsize=10)
        ax2.set_ylabel('Вероятность\nпоражения', fontsize=10)
        ax2.set_title('Зависимость вероятности поражения от расстояния',
                      fontsize=12, pad=20)
        ax2.legend(fontsize=10)

        # График пробит-функции
        ax3.plot(radius_arr, probit_arr, color=colors[2], linewidth=2.5,
                 label='Пробит-функция')
        ax3.set_xlabel('Расстояние, м', fontsize=10)
        ax3.set_ylabel('Значение\nпробит-функции', fontsize=10)
        ax3.set_title('Зависимость пробит-функции от расстояния',
                      fontsize=12, pad=20)
        ax3.legend(fontsize=10)

        # Добавляем подписи зон поражения на график интенсивности
        zone_colors = ['#FFE5E5', '#FFE5CC', '#FFFFCC', '#E5FFE5']  # Более светлые цвета для зон
        prev_x = 0

        for i, zone_value in enumerate(self.CLASSIFIED_ZONES):
            idx = q_term_arr.index(get_nearest_value(q_term_arr, zone_value))
            x = radius_arr[idx]

            # Закрашиваем зону
            ax1.axvspan(prev_x, x, alpha=0.3, color=zone_colors[i])

            # Добавляем вертикальную линию и подпись
            ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
            ax1.text(x, max(q_term_arr) * 0.7, f'{zone_value} кВт/м²',
                     rotation=90, verticalalignment='bottom',
                     horizontalalignment='right')
            prev_x = x

        # Добавляем описание зон внизу графика
        zone_descriptions = [
            '10.5 кВт/м² - непереносимая боль через 3-5 с',
            '7.0 кВт/м² - непереносимая боль через 20-30 с',
            '4.2 кВт/м² - безопасно в брезентовой одежде',
            '1.4 кВт/м² - безопасно для человека'
        ]

        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='gray', alpha=0.8)
        description_text = '\n'.join(zone_descriptions)
        fig.text(0.98, 0.02, description_text, fontsize=9,
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 bbox=bbox_props)

        # Настройка отступов
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)

        return fig

        # Цветовая схема
        colors = ['#2E86C1', '#E74C3C', '#27AE60']  # Синий, Красный, Зеленый

        # График интенсивности теплового излучения
        ax1.plot(radius_arr, q_term_arr, color=colors[0], linewidth=2.5,
                 label='Интенсивность теплового излучения')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Расстояние, м', fontsize=10)
        ax1.set_ylabel('Интенсивность,\nкВт/м²', fontsize=10)
        ax1.set_title('Зависимость интенсивности теплового излучения от расстояния',
                      fontsize=12, pad=20)
        ax1.legend(fontsize=10)

        # Добавляем исходные данные
        info_text = (
            f"Исходные параметры:\n\n"
            f"Площадь пролива: {params.S_spill} м²\n"
            f"Скорость выгорания: {params.m_sg} кг/(с·м²)\n"
            f"Молекулярная масса: {params.mol_mass} кг/кмоль\n"
            f"Температура кипения: {params.t_boiling} °C\n"
            f"Скорость ветра: {params.wind_velocity} м/с"
        )

        # Размещаем текст в рамке
        ax1.text(0.98, 0.98, info_text,
                 transform=ax1.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.7',
                           facecolor='white',
                           edgecolor='gray',
                           alpha=0.8),
                 fontsize=10)

        # График вероятности поражения
        ax2.plot(radius_arr, probability_arr, color=colors[1], linewidth=2.5,
                 label='Вероятность поражения')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Расстояние, м', fontsize=10)
        ax2.set_ylabel('Вероятность\nпоражения', fontsize=10)
        ax2.set_title('Зависимость вероятности поражения от расстояния',
                      fontsize=12, pad=20)
        ax2.legend(fontsize=10)

        # График пробит-функции
        ax3.plot(radius_arr, probit_arr, color=colors[2], linewidth=2.5,
                 label='Пробит-функция')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Расстояние, м', fontsize=10)
        ax3.set_ylabel('Значение\nпробит-функции', fontsize=10)
        ax3.set_title('Зависимость пробит-функции от расстояния',
                      fontsize=12, pad=20)
        ax3.legend(fontsize=10)

        # Добавляем подписи зон поражения на график интенсивности
        zone_colors = ['#FF9999', '#FFB366', '#FFFF99', '#99FF99']  # Цвета для зон
        prev_x = 0

        for i, zone_value in enumerate(self.CLASSIFIED_ZONES):
            idx = q_term_arr.index(get_nearest_value(q_term_arr, zone_value))
            x = radius_arr[idx]

            # Закрашиваем зону
            ax1.axvspan(prev_x, x, alpha=0.2, color=zone_colors[i])

            # Добавляем вертикальную линию и подпись
            ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
            ax1.text(x, max(q_term_arr) * 0.7, f'{zone_value} кВт/м²',
                     rotation=90, verticalalignment='bottom',
                     horizontalalignment='right')
            prev_x = x

        # Добавляем описание зон внизу графика
        zone_descriptions = [
            '10.5 кВт/м² - непереносимая боль через 3-5 с',
            '7.0 кВт/м² - непереносимая боль через 20-30 с',
            '4.2 кВт/м² - безопасно в брезентовой одежде',
            '1.4 кВт/м² - безопасно для человека'
        ]

        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor='gray', alpha=0.8)
        description_text = '\n'.join(zone_descriptions)
        fig.text(0.98, 0.02, description_text, fontsize=9,
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 bbox=bbox_props)

        # Настройка отступов
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)

        return fig


def main():
    """
    Пример использования класса
    """
    # Создание параметров расчета
    params = InputParameters(
        S_spill=200,  # площадь пролива, м2
        m_sg=0.06,  # удельная плотность выгорания, кг/(с*м2)
        mol_mass=100,  # молекулярная масса, кг/кмоль
        t_boiling=63,  # температура кипения, град.С
        wind_velocity=1  # скорость ветра, м/с
    )

    try:
        # Создание экземпляра класса
        fire_calc = StraitFire()

        # Расчет теплового излучения в конкретной точке
        radius = 20  # расстояние от центра пролива, м
        q_term = fire_calc.termal_radiation_point(params, radius)
        print(f"\nРезультаты расчета теплового излучения:")
        print("-" * 50)
        print(f"Интенсивность теплового излучения на расстоянии {radius}м: {q_term:.2f} кВт/м²")

        # Расчет массивов значений
        radius_arr, q_term_arr, probit_arr, probability_arr = fire_calc.termal_radiation_array(params)

        print("\nРезультаты расчета по расстояниям:")
        print("-" * 75)
        print("Расстояние (м) | Излучение (кВт/м²) | Пробит | Вероятность поражения")
        print("-" * 75)

        # Выводим первые 10 значений
        for r, q, pr, prob in zip(radius_arr[:10], q_term_arr[:10], probit_arr[:10], probability_arr[:10]):
            print(f"{r:12.2f} | {q:17.2f} | {pr:7.2f} | {prob:.3f}")
        print("...")

        # Расчет зон поражения
        zones = fire_calc.termal_class_zone(params)

        print("\nРадиусы зон поражения:")
        print("-" * 50)
        for zone_value, radius in zip(StraitFire.CLASSIFIED_ZONES, zones):
            print(f"Зона {zone_value:4.1f} кВт/м² - радиус {radius:.2f} м")

        # Дополнительная информация
        D_eff = fire_calc.calculate_effective_diameter(params.S_spill)
        print(f"\nДополнительная информация:")
        print("-" * 50)
        print(f"Эффективный диаметр пролива: {D_eff:.2f} м")
        print(f"Площадь пролива: {params.S_spill:.2f} м²")

        # Построение графиков
        fig = fire_calc.plot_results(params)
        # Сохраняем графики в файл вместо показа
        plt.savefig('strait_fire_results.png', dpi=300, bbox_inches='tight')
        print("\nГрафики сохранены в файл 'strait_fire_results.png'")
        plt.close()

    except ValueError as e:
        print(f"Ошибка в расчетах: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")


if __name__ == '__main__':
    main()