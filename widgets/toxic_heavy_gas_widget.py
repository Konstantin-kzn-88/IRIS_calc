# widgets/toxic_dispersion_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout, QRadioButton, QButtonGroup
from widgets.base_calculator import BaseCalculatorWidget
from models.Toxic_dispersions.calc_heavy_gas_disp import Instantaneous_source, Continuous_source


class ToxicDispersionCalculatorWidget(BaseCalculatorWidget):
    """Виджет для расчета рассеяния тяжелого газа"""

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выбор типа выброса
        release_layout = QHBoxLayout()
        release_label = QLabel("Тип выброса:")
        self.instantaneous_radio = QRadioButton("Мгновенный")
        self.continuous_radio = QRadioButton("Постоянный")
        self.instantaneous_radio.setChecked(True)

        self.release_type_group = QButtonGroup()
        self.release_type_group.addButton(self.instantaneous_radio)
        self.release_type_group.addButton(self.continuous_radio)

        release_layout.addWidget(release_label)
        release_layout.addWidget(self.instantaneous_radio)
        release_layout.addWidget(self.continuous_radio)
        release_layout.addStretch()
        self.input_layout.addRow(release_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Общие параметры
        for param, (label, value, decimals) in {
            'wind_speed': ('Скорость ветра [м/с]', 1.0, 1),
            'density_air': ('Плотность воздуха [кг/м³]', 1.21, 2),
            'density_init': ('Плотность газа [кг/м³]', 6.0, 2),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.01, 1000)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

        # Параметры мгновенного выброса
        self.inputs['volume_gas'] = QDoubleSpinBox()
        self.inputs['volume_gas'].setRange(0.1, 10000)
        self.inputs['volume_gas'].setValue(10.0)
        self.inputs['volume_gas'].setDecimals(1)
        self.input_layout.addRow("Объем газа [м³]:", self.inputs['volume_gas'])

        # Параметры постоянного выброса
        self.inputs['gas_flow'] = QDoubleSpinBox()
        self.inputs['gas_flow'].setRange(0.01, 1000)
        self.inputs['gas_flow'].setValue(1.0)
        self.inputs['gas_flow'].setDecimals(2)
        self.input_layout.addRow("Расход газа [кг/с]:", self.inputs['gas_flow'])

        self.inputs['radius_flow'] = QDoubleSpinBox()
        self.inputs['radius_flow'].setRange(0.01, 10)
        self.inputs['radius_flow'].setValue(0.1)
        self.inputs['radius_flow'].setDecimals(2)
        self.input_layout.addRow("Радиус выброса [м]:", self.inputs['radius_flow'])

        # Добавляем поля для пороговых значений токсодоз
        self.inputs['threshold_dose'] = QDoubleSpinBox()
        self.inputs['threshold_dose'].setRange(0.1, 1000)
        self.inputs['threshold_dose'].setValue(15.0)
        self.inputs['threshold_dose'].setDecimals(1)
        self.input_layout.addRow("Пороговая токсодоза [мг·мин/л]:", self.inputs['threshold_dose'])

        self.inputs['lethal_dose'] = QDoubleSpinBox()
        self.inputs['lethal_dose'].setRange(0.1, 1000)
        self.inputs['lethal_dose'].setValue(60.0)
        self.inputs['lethal_dose'].setDecimals(1)
        self.input_layout.addRow("Смертельная токсодоза [мг·мин/л]:", self.inputs['lethal_dose'])

        # Подключаем обработчик изменения типа выброса
        self.release_type_group.buttonClicked.connect(self.on_release_type_changed)
        self.on_release_type_changed()

    def on_release_type_changed(self):
        """Обработка изменения типа выброса"""
        is_instantaneous = self.instantaneous_radio.isChecked()
        self.inputs['volume_gas'].setEnabled(is_instantaneous)
        self.inputs['gas_flow'].setEnabled(not is_instantaneous)
        self.inputs['radius_flow'].setEnabled(not is_instantaneous)

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def find_threshold_distances(self, toxdoses, distances, threshold_dose, lethal_dose):
        """Поиск расстояний для пороговых значений токсодоз"""
        threshold_dist = None
        lethal_dist = None

        for i in range(len(toxdoses) - 1):
            if toxdoses[i] >= threshold_dose >= toxdoses[i + 1]:
                ratio = (threshold_dose - toxdoses[i + 1]) / (toxdoses[i] - toxdoses[i + 1])
                threshold_dist = distances[i + 1] + ratio * (distances[i] - distances[i + 1])
            if toxdoses[i] >= lethal_dose >= toxdoses[i + 1]:
                ratio = (lethal_dose - toxdoses[i + 1]) / (toxdoses[i] - toxdoses[i + 1])
                lethal_dist = distances[i + 1] + ratio * (distances[i] - distances[i + 1])

        return threshold_dist, lethal_dist

    def plot_results(self, results, params, threshold_dist, lethal_dist):
        """Построение графиков результатов"""
        axes = self.figure.subplots(2, 1)

        # График концентраций
        axes[0].plot(results[2], results[1], 'b-', linewidth=2)
        axes[0].set_xlabel('Расстояние, м')
        axes[0].set_ylabel('Концентрация, кг/м³')
        axes[0].grid(True)
        axes[0].set_title('Распределение концентрации')

        # График токсодоз
        axes[1].plot(results[2], results[0], 'r-', linewidth=2)
        axes[1].set_xlabel('Расстояние, м')
        axes[1].set_ylabel('Токсодоза, мг·мин/л')
        axes[1].grid(True)
        axes[1].set_title('Распределение токсодоз')

        # Добавляем линии пороговых значений
        if threshold_dist:
            axes[1].axhline(y=params['threshold_dose'], color='orange', linestyle='--', alpha=0.5)
            axes[1].axvline(x=threshold_dist, color='orange', linestyle='--', alpha=0.5)
            axes[1].text(threshold_dist, max(results[0]) * 0.8,
                         f'Пороговая зона\n{threshold_dist:.1f} м',
                         rotation=90, verticalalignment='top')

        if lethal_dist:
            axes[1].axhline(y=params['lethal_dose'], color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(x=lethal_dist, color='red', linestyle='--', alpha=0.5)
            axes[1].text(lethal_dist, max(results[0]) * 0.8,
                         f'Смертельная зона\n{lethal_dist:.1f} м',
                         rotation=90, verticalalignment='top')

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Создание модели в зависимости от типа выброса
            if self.instantaneous_radio.isChecked():
                model = Instantaneous_source(
                    wind_speed=params['wind_speed'],
                    density_air=params['density_air'],
                    density_init=params['density_init'],
                    volume_gas=params['volume_gas']
                )
            else:
                model = Continuous_source(
                    wind_speed=params['wind_speed'],
                    density_air=params['density_air'],
                    density_init=params['density_init'],
                    gas_flow=params['gas_flow'],
                    radius_flow=params['radius_flow']
                )

            # Получение результатов
            results = model.result()

            # Поиск расстояний для пороговых значений
            threshold_dist, lethal_dist = self.find_threshold_distances(
                results[0], results[2],
                params['threshold_dose'],
                params['lethal_dose']
            )

            # Вывод результатов
            release_type = "мгновенный" if self.instantaneous_radio.isChecked() else "постоянный"
            self.output_text.append(f"Результаты расчета ({release_type} выброс):")
            self.output_text.append("-" * 50)

            if threshold_dist:
                self.output_text.append(f"Радиус пороговой зоны: {threshold_dist:.1f} м")
            if lethal_dist:
                self.output_text.append(f"Радиус смертельной зоны: {lethal_dist:.1f} м")

            self.output_text.append("-" * 30)
            for i in range(len(results[0])):
                self.output_text.append(
                    f"Токсодоза: {results[0][i]:.2f} мг·мин/л\n"
                    f"Концентрация: {results[1][i]:.3f} кг/м³\n"
                    f"Расстояние: {results[2][i]:.1f} м"
                )
                if self.instantaneous_radio.isChecked():
                    self.output_text.append(
                        f"Ширина облака: {results[3][i]:.1f} м\n"
                        f"Время подхода: {results[4][i]:.1f} с"
                    )
                else:
                    self.output_text.append(f"Ширина облака: {results[3][i]:.1f} м")
                self.output_text.append("-" * 30)

            # Построение графиков
            self.plot_results(results, params, threshold_dist, lethal_dist)

            # Добавляем информацию о параметрах
            info_text = (
                f"Параметры расчета:\n"
                f"Скорость ветра: {params['wind_speed']} м/с\n"
                f"Плотность воздуха: {params['density_air']} кг/м³\n"
                f"Плотность газа: {params['density_init']} кг/м³\n"
            )

            if self.instantaneous_radio.isChecked():
                info_text += f"Объем газа: {params['volume_gas']} м³"
            else:
                info_text += f"Расход газа: {params['gas_flow']} кг/с\n"
                info_text += f"Радиус выброса: {params['radius_flow']} м"

            self.figure.text(0.02, 0.02, info_text, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.8))

            # Обновляем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")