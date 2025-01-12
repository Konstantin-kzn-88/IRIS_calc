from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QFormLayout, QPushButton, QTextEdit)
from abc import ABC, abstractmethod
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BaseCalculatorWidget(QWidget):
    """Базовый класс для всех виджетов расчета"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Настройка базового интерфейса"""
        # Основной горизонтальный layout
        self.main_layout = QHBoxLayout(self)

        # Левая панель для ввода и вывода текста
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Группа для ввода параметров
        self.input_group = QGroupBox("Исходные данные")
        self.input_layout = QFormLayout(self.input_group)
        self.setup_inputs()
        left_layout.addWidget(self.input_group)

        # Кнопка расчета
        self.calculate_button = QPushButton("Выполнить расчет")
        self.calculate_button.clicked.connect(self.on_calculate)
        left_layout.addWidget(self.calculate_button)

        # Группа для вывода результатов
        self.output_group = QGroupBox("Результаты расчета")
        self.output_layout = QVBoxLayout(self.output_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_layout.addWidget(self.output_text)
        left_layout.addWidget(self.output_group)

        # Правая панель для графиков
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Создание области для графиков
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        # Добавляем панели в основной layout с нужными пропорциями
        self.main_layout.addWidget(left_panel, 30)  # 30% ширины
        self.main_layout.addWidget(right_panel, 70)  # 70% ширины

    @abstractmethod
    def setup_inputs(self):
        """Метод для настройки специфичных полей ввода"""
        pass

    @abstractmethod
    def get_params(self):
        """Метод для получения параметров из полей ввода"""
        pass

    @abstractmethod
    def on_calculate(self):
        """Метод для выполнения расчетов"""
        pass


class BaseCalculatorTab(QWidget):
    """Базовый класс для вкладки с калькулятором"""

    def __init__(self, calculator_widget_class, parent=None):
        super().__init__(parent)
        self.setup_ui(calculator_widget_class)

    def setup_ui(self, calculator_widget_class):
        """Настройка интерфейса вкладки"""
        layout = QVBoxLayout(self)
        self.calculator = calculator_widget_class(self)
        layout.addWidget(self.calculator)