import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class EvacuationPath:
    """Класс для представления участка пути эвакуации"""
    length: float  # длина участка в метрах
    width: float  # ширина участка в метрах
    type: str  # тип участка (horizontal, door, stairs_down, stairs_up)
    people_count: int  # количество людей на участке


class EvacuationCalculator:
    """Калькулятор времени эвакуации"""

    def __init__(self):
        # Константы из таблицы П5.1
        self.movement_parameters = {
            'horizontal': {
                'density': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'velocity': [100, 100, 80, 60, 47, 40, 33, 28, 23, 19, 15],
                'intensity': [1.0, 5.0, 8.0, 12.0, 14.1, 16.0, 16.5, 16.3, 16.1, 15.2, 13.5]
            },
            'door': {
                'density': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'intensity': [1.0, 5.0, 8.7, 13.4, 16.5, 18.4, 19.6, 19.05, 18.5, 17.3, 8.5]
            },
            'stairs_down': {
                'density': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'velocity': [100, 100, 95, 68, 52, 40, 31, 24.5, 18, 13, 8],
                'intensity': [1.0, 5.0, 9.5, 13.6, 15.6, 16.0, 15.6, 14.1, 12.6, 10.4, 7.2]
            },
            'stairs_up': {
                'density': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'velocity': [60, 60, 53, 40, 32, 26, 22, 18.5, 15, 13, 11],
                'intensity': [0.6, 3.0, 5.3, 8.0, 9.6, 10.4, 11.0, 10.75, 10.5, 10.4, 9.9]
            }
        }

    def calculate_density(self, path: EvacuationPath) -> float:
        """Расчет плотности людского потока на участке"""
        if path.type == 'door':
            # Для дверных проемов используем минимальную плотность
            return 0.01
        area = path.length * path.width
        person_area = 0.125  # средняя площадь горизонтальной проекции человека
        return (path.people_count * person_area) / area

    def get_movement_parameters(self, path_type: str, density: float) -> Tuple[Optional[float], float]:
        """Получение параметров движения для заданной плотности потока"""
        params = self.movement_parameters[path_type]

        # Находим индекс для интерполяции
        for i in range(len(params['density']) - 1):
            if params['density'][i] <= density <= params['density'][i + 1]:
                factor = (density - params['density'][i]) / (params['density'][i + 1] - params['density'][i])

                intensity = params['intensity'][i] + factor * (params['intensity'][i + 1] - params['intensity'][i])

                # Скорость есть не для всех типов путей
                velocity = None
                if 'velocity' in params:
                    velocity = params['velocity'][i] + factor * (params['velocity'][i + 1] - params['velocity'][i])

                return velocity, intensity

        # Если плотность выше максимальной
        if density > params['density'][-1]:
            velocity = params['velocity'][-1] if 'velocity' in params else None
            return velocity, params['intensity'][-1]

        # Если плотность ниже минимальной
        velocity = params['velocity'][0] if 'velocity' in params else None
        return velocity, params['intensity'][0]

    def calculate_movement_time(self, path: EvacuationPath) -> float:
        """Расчет времени движения по участку пути"""
        if path.type == 'door':
            # Для дверных проемов рассчитываем время прохождения через проем
            if path.width >= 1.6:
                q_door = 8.5  # максимальная интенсивность для широких проемов
            else:
                q_door = 2.5 + 3.75 * path.width  # формула для узких проемов
            return path.people_count / (q_door * path.width * 60)  # переводим в минуты

        density = self.calculate_density(path)
        velocity, _ = self.get_movement_parameters(path.type, density)

        if velocity is None:
            raise ValueError(f"Не удалось определить скорость для участка типа {path.type}")

        # Время в минутах
        return path.length / velocity

    def calculate_total_evacuation_time(self, paths: List[EvacuationPath]) -> float:
        """Расчет общего времени эвакуации"""
        total_time = 0

        for i, path in enumerate(paths):
            # Расчет времени движения по текущему участку
            movement_time = self.calculate_movement_time(path)
            total_time += movement_time

            # Проверка на образование скоплений людей
            if i < len(paths) - 1:  # если это не последний участок
                next_path = paths[i + 1]

                # Проверка условия образования скопления
# Проверка условия образования скопления
                current_density = self.calculate_density(path)
                _, current_intensity = self.get_movement_parameters(path.type, current_density)
                current_flow = current_intensity * path.width

                # Максимальная интенсивность для следующего участка
                if next_path.type == 'door':
                    max_next_intensity = 8.5 if next_path.width >= 1.6 else (2.5 + 3.75 * next_path.width)
                else:
                    max_next_intensity = 16.5

                max_next_flow = max_next_intensity * next_path.width

                if current_flow > max_next_flow:
                    # Добавляем время задержки
                    delay_time = self.calculate_delay_time(path, next_path, current_flow, max_next_flow)
                    total_time += delay_time

            print(f"Участок {i + 1} ({path.type}): {movement_time:.2f} мин")

        return total_time

    def calculate_delay_time(self, current_path: EvacuationPath, next_path: EvacuationPath,
                             current_flow: float, max_next_flow: float) -> float:
        """Расчет времени задержки при образовании скопления"""
        person_area = 0.125
        N = current_path.people_count

        delay_time = (N * person_area) * (1 / max_next_flow - 1 / current_flow)
        return max(0, delay_time)


def main():
    # Создаем калькулятор эвакуации
    evacuation_calc = EvacuationCalculator()

    # Создаем пути эвакуации
    paths = [
        # Горизонтальный участок (коридор)
        EvacuationPath(length=20, width=2, type='horizontal', people_count=50),
        # Дверной проем
        EvacuationPath(length=0, width=1.2, type='door', people_count=50),
        # Лестница вниз
        EvacuationPath(length=15, width=1.5, type='stairs_down', people_count=50),
        # Горизонтальный участок (холл)
        EvacuationPath(length=10, width=3, type='horizontal', people_count=50),
        # Выходная дверь
        EvacuationPath(length=0, width=1.6, type='door', people_count=50)
    ]

    # Рассчитываем время эвакуации
    total_time = evacuation_calc.calculate_total_evacuation_time(paths)
    print(f"\nОбщее расчетное время эвакуации: {total_time:.2f} минут")




if __name__ == "__main__":
    main()