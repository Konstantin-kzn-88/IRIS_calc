# widgets/instant_destruction_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QProgressBar
from PySide6.QtCore import Qt, QThread, Signal
from widgets.base_calculator import BaseCalculatorWidget
from models.Destruction_tank.instant_destruction import LiquidSpreadModel
import numpy as np


class CalculationThread(QThread):
    """Поток для выполнения расчетов"""
    finished = Signal(tuple)  # Сигнал с результатами
    progress = Signal(int)  # Сигнал для обновления прогресса
    error = Signal(str)  # Сигнал для ошибок

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self):
        try:
            # Создание модели
            self.progress.emit(10)

            model = LiquidSpreadModel(
                R=self.params['R'],
                h0=self.params['h0'],
                a=self.params['a'],
                b=self.params['b'],
                dx=self.params['dx']
            )

            self.progress.emit(20)

            # Решение уравнений
            t, h, u, overflow_history = model.solve(self.params['T'])

            self.progress.emit(90)

            # Анализ результатов
            max_height = np.max(h)
            max_height_time = t[np.argmax(np.max(h, axis=1))]
            max_overflow = np.max(overflow_history)

            # Отправляем результаты через сигнал
            self.finished.emit((t, h, u, overflow_history, max_height, max_height_time, max_overflow))
            self.progress.emit(100)

        except Exception as e:
            self.error.emit(str(e))


class InstantDestructionWidget(BaseCalculatorWidget):
    """Виджет для расчета перелива жидкости через обвалование"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем поля ввода
        self.inputs = {}

        # Параметры резервуара и обвалования
        for param, (label, value, decimals) in {
            'R': ('Ширина резервуара [м]', 5.0, 1),
            'h0': ('Начальная высота жидкости [м]', 6.0, 1),
            'a': ('Высота обвалования [м]', 1.0, 1),
            'b': ('Расстояние до обвалования [м]', 10.0, 1),
            'dx': ('Шаг по пространству [м]', 0.1, 2),
            'T': ('Время моделирования [с]', 5.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.01, 1000.0)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Добавляем прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.input_layout.addRow("Прогресс расчета:", self.progress_bar)

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def update_progress(self, value):
        """Обновление прогресс-бара"""
        self.progress_bar.setValue(value)

    def calculation_error(self, error_msg):
        """Обработка ошибок расчета"""
        self.output_text.setText(f"Ошибка при расчете: {error_msg}")
        self.progress_bar.setVisible(False)
        self.calculate_button.setEnabled(True)

    def calculation_finished(self, results):
        """Обработка завершения расчета"""
        t, h, u, overflow_history, max_height, max_height_time, max_overflow = results

        try:
            # Вывод результатов
            self.output_text.clear()
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(f"Максимальная высота волны: {max_height:.2f} м")
            # self.output_text.append(f"Время достижения максимума: {max_height_time:.2f} с")
            self.output_text.append(f"Максимальный перелив: {max_overflow:.2f}%")

            # Получаем параметры для построения графика
            params = self.get_params()

            # Очищаем фигуру
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # График высоты жидкости в разные моменты времени
            times_to_plot = np.linspace(0, len(t) - 1, 4).astype(int)
            x = np.arange(0, 1.5 * params['b'], params['dx'])
            barrier_x = params['b']

            for i in times_to_plot:
                ax.plot(x, h[i], label=f't = {t[i]:.2f} с')

            # Добавляем обвалование
            ax.vlines(barrier_x, 0, params['a'], colors='r', linestyles='--', label='Обвалование')
            ax.hlines(params['a'], barrier_x - 0.2, barrier_x + 0.2, colors='r', linestyles='--')

            ax.set_xlabel('Расстояние [м]')
            ax.set_ylabel('Высота жидкости [м]')
            ax.set_title('Профиль высоты жидкости')
            ax.grid(True)
            ax.legend()

            # Добавляем информацию о параметрах и результатах
            info_text = (
                f"Параметры расчета:\n"
                f"Ширина резервуара: {params['R']:.1f} м\n"
                f"Начальная высота: {params['h0']:.1f} м\n"
                f"Высота обвалования: {params['a']:.1f} м\n"
                f"Расстояние до обвалования: {params['b']:.1f} м\n\n"
                f"Результаты:\n"
                f"Макс. высота волны: {max_height:.2f} м\n"
                f"Время макс. высоты: {max_height_time:.2f} с\n"
                f"Макс. перелив: {max_overflow:.2f}%"
            )
            ax.text(0.02, 0.98, info_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=8)

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

            # Завершаем процесс
            self.progress_bar.setVisible(False)
            self.calculate_button.setEnabled(True)

        except Exception as e:
            self.calculation_error(str(e))

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Показываем прогресс-бар и блокируем кнопку
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.calculate_button.setEnabled(False)

            # Получение параметров
            params = self.get_params()

            # Создаем поток для расчетов
            self.calc_thread = CalculationThread(params)
            self.calc_thread.progress.connect(self.update_progress)
            self.calc_thread.finished.connect(self.calculation_finished)
            self.calc_thread.error.connect(self.calculation_error)

            # Запускаем расчет
            self.calc_thread.start()

        except Exception as e:
            self.calculation_error(str(e))