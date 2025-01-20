from PySide6.QtWidgets import (QDoubleSpinBox, QSpinBox, QComboBox,
                             QLabel, QHBoxLayout, QCheckBox)
from widgets.base_calculator import BaseCalculatorWidget
from models.Toxic_dispersions.calc_light_gas_disp import Source


class ToxicCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета токсического поражения"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Создаем поля ввода
        self.inputs = {}

        # Параметры окружающей среды
        for param, (label, value, decimals) in {
            'ambient_temperature': ('Температура воздуха [°C]', 20.0, 1),
            'cloud': ('Облачность [0-8]', 0.0, 0),
            'wind_speed': ('Скорость ветра [м/с]', 1.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(-50 if param == 'ambient_temperature' else 0,
                                      50 if param == 'ambient_temperature' else
                                      8 if param == 'cloud' else 30)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Чекбоксы для условий
        checkbox_layout = QHBoxLayout()
        self.is_night = QCheckBox("Ночное время")
        self.is_urban = QCheckBox("Городская застройка")
        checkbox_layout.addWidget(self.is_night)
        checkbox_layout.addWidget(self.is_urban)
        self.input_layout.addRow(checkbox_layout)

        # Параметры выброса
        for param, (label, value, decimals) in {
            'ejection_height': ('Высота выброса [м]', 2.0, 1),
            'gas_temperature': ('Температура газа [°C]', 20.0, 1),
            'gas_weight': ('Масса газа [кг]', 0.0, 1),
            'gas_flow': ('Расход газа [кг/с]', 0.056, 3),
            'closing_time': ('Время отсечения [с]', 300.0, 0),
            'molecular_weight': ('Молекулярная масса [кг/кмоль]', 17.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 10000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Пороговые значения токсодоз
        for param, (label, value, decimals) in {
            'threshold_dose': ('Пороговая токсодоза [мг·мин/л]', 15.0, 1),
            'lethal_dose': ('Смертельная токсодоза [мг·мин/л]', 60.0, 1),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0, 1000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        params = {key: widget.value() for key, widget in self.inputs.items()}
        params['is_night'] = int(self.is_night.isChecked())
        params['is_urban_area'] = int(self.is_urban.isChecked())
        return params

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Создание модели и выполнение расчетов
            source = Source(
                ambient_temperature=params['ambient_temperature'],
                cloud=params['cloud'],
                wind_speed=params['wind_speed'],
                is_night=params['is_night'],
                is_urban_area=params['is_urban_area'],
                ejection_height=params['ejection_height'],
                gas_temperature=params['gas_temperature'],
                gas_weight=params['gas_weight'],
                gas_flow=params['gas_flow'],
                closing_time=params['closing_time'],
                molecular_weight=params['molecular_weight']
            )

            # Расчет зон поражения
            threshold_dist, lethal_dist = source.get_threshold_distances(
                threshold_dose=params['threshold_dose'],
                lethal_dose=params['lethal_dose']
            )

            # Получение массивов данных для построения графиков
            distances, concentrations, doses = source.result()

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(
                f"Класс стабильности атмосферы: {source.pasquill}")
            self.output_text.append(
                f"Плотность газа: {source.gas_density:.3f} кг/м³")
            self.output_text.append(
                f"Радиус первичного облака: {source.radius_first_cloud:.1f} м")
            self.output_text.append("\nЗоны поражения:")
            self.output_text.append(
                f"Пороговая зона: {threshold_dist:.1f} м")
            self.output_text.append(
                f"Смертельная зона: {lethal_dist:.1f} м")

            # Построение графиков
            ((ax1, ax2)) = self.figure.subplots(2, 1)

            # График концентрации
            ax1.plot(distances, concentrations, 'b-', linewidth=2)
            ax1.set_xlabel('Расстояние, м')
            ax1.set_ylabel('Концентрация, кг/м³')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_title('Распределение концентрации')

            # График токсодозы
            ax2.plot(distances, doses, 'r-', linewidth=2)
            ax2.set_xlabel('Расстояние, м')
            ax2.set_ylabel('Токсодоза, мг·мин/л')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_title('Распределение токсодозы')

            # Добавляем вертикальные линии зон поражения
            for dist, label, color in [
                (threshold_dist, f'Пороговая зона\n{threshold_dist:.1f} м', 'orange'),
                (lethal_dist, f'Смертельная зона\n{lethal_dist:.1f} м', 'red')
            ]:
                ax1.axvline(x=dist, color=color, linestyle='--', alpha=0.5)
                ax2.axvline(x=dist, color=color, linestyle='--', alpha=0.5)
                ax2.text(dist, max(doses) * 0.8, label,
                         rotation=90, verticalalignment='top')

            # Добавляем исходные данные
            info_text = (
                f"Исходные параметры:\n"
                f"Температура воздуха: {params['ambient_temperature']:.1f}°C\n"
                f"Скорость ветра: {params['wind_speed']:.1f} м/с\n"
                f"Облачность: {params['cloud']:.0f}\n"
                f"Время суток: {'ночь' if params['is_night'] else 'день'}\n"
                f"Застройка: {'городская' if params['is_urban_area'] else 'открытая'}\n"
                f"Высота выброса: {params['ejection_height']:.1f} м\n"
                f"Температура газа: {params['gas_temperature']:.1f}°C"
            )
            ax1.text(0.02, 0.98, info_text,
                     transform=ax1.transAxes,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5',
                              facecolor='white',
                              alpha=0.8),
                     fontsize=8)

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")