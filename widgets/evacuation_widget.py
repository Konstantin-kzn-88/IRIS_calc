# widgets/evacuation_widget.py
from PySide6.QtWidgets import (QDoubleSpinBox, QSpinBox, QComboBox,
                               QLabel, QHBoxLayout, QPushButton, QVBoxLayout,
                               QGridLayout, QFrame)
from widgets.base_calculator import BaseCalculatorWidget
from models.Evacuation.routes import EvacuationCalculator, EvacuationPath
import numpy as np


class EvacuationCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета времени эвакуации"""

    PATH_TYPES = {
        'horizontal': 'Горизонтальный участок',
        'door': 'Дверной проем',
        'stairs_down': 'Лестница вниз',
        'stairs_up': 'Лестница вверх'
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем контейнер для путей эвакуации
        self.paths_container = QVBoxLayout()
        self.paths = []
        self.path_widgets = []

        # Кнопка добавления нового пути
        add_path_button = QPushButton("Добавить участок пути")
        add_path_button.clicked.connect(self.add_path_widget)
        self.input_layout.addRow(add_path_button)

        # Добавляем контейнер путей в основной layout
        self.input_layout.addRow(self.paths_container)

        # Добавляем первый путь по умолчанию
        self.add_path_widget()

    def create_path_widget(self):
        """Создание виджета для одного участка пути"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QGridLayout()
        frame.setLayout(layout)

        # Тип участка
        type_combo = QComboBox()
        for type_id, type_name in self.PATH_TYPES.items():
            type_combo.addItem(type_name, type_id)
        layout.addWidget(QLabel("Тип участка:"), 0, 0)
        layout.addWidget(type_combo, 0, 1)

        # Длина участка
        length_spin = QDoubleSpinBox()
        length_spin.setRange(0, 1000)
        length_spin.setValue(10)
        length_spin.setSuffix(" м")
        layout.addWidget(QLabel("Длина:"), 1, 0)
        layout.addWidget(length_spin, 1, 1)

        # Ширина участка
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.1, 100)
        width_spin.setValue(1.5)
        width_spin.setSuffix(" м")
        layout.addWidget(QLabel("Ширина:"), 1, 2)
        layout.addWidget(width_spin, 1, 3)

        # Количество людей
        people_spin = QSpinBox()
        people_spin.setRange(1, 1000)
        people_spin.setValue(50)
        people_spin.setSuffix(" чел.")
        layout.addWidget(QLabel("Количество людей:"), 2, 0)
        layout.addWidget(people_spin, 2, 1)

        # Кнопка удаления
        delete_button = QPushButton("Удалить участок")
        delete_button.clicked.connect(lambda: self.remove_path_widget(frame))
        layout.addWidget(delete_button, 2, 3)

        return frame, {
            'type': type_combo,
            'length': length_spin,
            'width': width_spin,
            'people': people_spin
        }

    def add_path_widget(self):
        """Добавление нового виджета пути"""
        frame, widgets = self.create_path_widget()
        self.paths_container.addWidget(frame)
        self.path_widgets.append((frame, widgets))

    def remove_path_widget(self, frame):
        """Удаление виджета пути"""
        if len(self.path_widgets) > 1:  # оставляем хотя бы один путь
            frame.deleteLater()
            self.path_widgets = [(f, w) for f, w in self.path_widgets if f != frame]

    def get_params(self):
        """Получение параметров из полей ввода"""
        paths = []
        for _, widgets in self.path_widgets:
            paths.append(EvacuationPath(
                type=widgets['type'].currentData(),
                length=widgets['length'].value(),
                width=widgets['width'].value(),
                people_count=widgets['people'].value()
            ))
        return paths

    def plot_results(self, calculator, paths, total_time):
        """Построение графиков результатов"""
        self.figure.clear()

        # Создаем три подграфика
        (ax1, ax2, ax3) = self.figure.subplots(3, 1)

        # График движения людей по участкам
        positions = []
        times = []
        labels = []
        cumulative_time = 0

        for i, path in enumerate(paths):
            movement_time = calculator.calculate_movement_time(path)
            positions.append(i)
            times.append(movement_time)
            labels.append(f"{self.PATH_TYPES[path.type]}\n{path.length}м")
            cumulative_time += movement_time

        ax1.bar(positions, times)
        ax1.set_xticks(positions)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Время прохождения (мин)')
        ax1.set_title('Время прохождения по участкам')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # График плотности людского потока
        densities = [calculator.calculate_density(path) for path in paths]
        ax2.bar(positions, densities)
        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Плотность (чел/м²)')
        ax2.set_title('Плотность людского потока')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # График движения людей во времени
        time_points = np.linspace(0, total_time, 100)
        people_remaining = []
        initial_people = paths[0].people_count

        for t in time_points:
            if t < total_time:
                fraction_complete = t / total_time
                people_left = initial_people * (1 - fraction_complete)
            else:
                people_left = 0
            people_remaining.append(people_left)

        ax3.plot(time_points, people_remaining, 'b-', linewidth=2)
        ax3.set_xlabel('Время (мин)')
        ax3.set_ylabel('Количество людей')
        ax3.set_title('Динамика эвакуации')
        ax3.grid(True, linestyle='--', alpha=0.7)

        # Добавляем информацию о параметрах
        info_text = (
            f"Общее время эвакуации: {total_time:.2f} мин\n"
            f"Количество участков: {len(paths)}\n"
            f"Общее количество людей: {paths[0].people_count}"
        )
        self.figure.text(0.02, 0.02, info_text, fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.8))

        self.figure.tight_layout()
        self.canvas.draw()

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()

            # Получение путей эвакуации
            paths = self.get_params()

            # Создание калькулятора и выполнение расчетов
            calculator = EvacuationCalculator()
            total_time = calculator.calculate_total_evacuation_time(paths)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)

            for i, path in enumerate(paths, 1):
                movement_time = calculator.calculate_movement_time(path)
                density = calculator.calculate_density(path)
                self.output_text.append(f"\nУчасток {i} ({self.PATH_TYPES[path.type]}):")
                self.output_text.append(f"Длина: {path.length:.1f} м")
                self.output_text.append(f"Ширина: {path.width:.1f} м")
                self.output_text.append(f"Количество людей: {path.people_count}")
                self.output_text.append(f"Плотность потока: {density:.2f} чел/м²")
                self.output_text.append(f"Время прохождения: {movement_time:.2f} мин")

            self.output_text.append(f"\nОбщее время эвакуации: {total_time:.2f} мин")

            # Построение графиков
            self.plot_results(calculator, paths, total_time)

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")