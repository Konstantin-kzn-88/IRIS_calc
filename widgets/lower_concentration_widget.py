# widgets/lower_concentration_widget.py
from PySide6.QtWidgets import QDoubleSpinBox, QComboBox, QLabel, QHBoxLayout
import numpy as np
from widgets.base_calculator import BaseCalculatorWidget
from models.LCLP.lower_concentration import LCLP


class LowerConcentrationWidget(BaseCalculatorWidget):
    """Виджет для расчета зон НКПР и пожара-вспышки для паров ЛВЖ"""

    # Предустановки для разных веществ
    PRESETS = {
        'gasoline': {
            'name': 'Бензин',
            'molecular_weight': 95.3,
            't_boiling': 100,
            'lower_concentration': 1.1,
        },
        'propane': {
            'name': 'Пропан',
            'molecular_weight': 44.1,
            't_boiling': -42.1,
            'lower_concentration': 2.1,
        },
        'pentane': {
            'name': 'Пентан',
            'molecular_weight': 72.15,
            't_boiling': 36.1,
            'lower_concentration': 1.4,
        }
    }

    def setup_inputs(self):
        """Настройка полей ввода"""
        # Выпадающий список для выбора предустановок
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Тип вещества:")
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Выберите тип вещества...")
        for preset_id, preset_data in self.PRESETS.items():
            self.preset_combo.addItem(preset_data['name'], preset_id)
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addStretch()
        self.input_layout.addRow(preset_layout)

        # Создаем поля ввода
        self.inputs = {}

        # Параметры вещества
        for param, (label, value, decimals) in {
            'mass': ('Масса паров [кг]', 10.0, 2),
            'molecular_weight': ('Молекулярная масса [кг/кмоль]', 95.3, 1),
            't_boiling': ('Температура кипения [°C]', 100.0, 1),
            'lower_concentration': ('Нижний концентрационный предел [% об.]', 1.1, 2),
        }.items():
            self.inputs[param] = QDoubleSpinBox()
            self.inputs[param].setRange(0.01, 1000.0)
            self.inputs[param].setValue(value)
            self.inputs[param].setDecimals(decimals)
            self.input_layout.addRow(label, self.inputs[param])

    def apply_preset(self, index):
        """Применение предустановки"""
        if index <= 0:
            return

        preset_id = self.preset_combo.currentData()
        preset = self.PRESETS[preset_id]

        # Обновляем значения в полях ввода
        for param in ['molecular_weight', 't_boiling', 'lower_concentration']:
            if param in preset:
                self.inputs[param].setValue(preset[param])

    def get_params(self):
        """Получение параметров из полей ввода"""
        return {key: widget.value() for key, widget in self.inputs.items()}

    def on_calculate(self):
        """Выполнение расчета"""
        try:
            # Очистка предыдущих результатов
            self.output_text.clear()
            self.figure.clear()

            # Получение параметров
            params = self.get_params()

            # Создание модели и выполнение расчета
            calculator = LCLP()
            result = calculator.lower_concentration_limit(**params)

            # Вывод результатов
            self.output_text.append("Результаты расчета:")
            self.output_text.append("-" * 50)
            self.output_text.append(f"Радиус НКПР: {result.r_lclp:.2f} м")
            self.output_text.append(f"Радиус пожара-вспышки: {result.r_flash:.2f} м")

            # Создаем подграфики
            (ax1, ax2) = self.figure.subplots(2, 1)

            # Массивы для построения графиков
            masses = np.linspace(1, params['mass'] * 2, 50)
            concentrations = np.linspace(0.5, params['lower_concentration'] * 2, 50)

            # Массивы для значений радиусов
            r_lclp_mass = []
            r_flash_mass = []
            r_lclp_conc = []
            r_flash_conc = []

            # Расчет значений в зависимости от массы
            for m in masses:
                result = calculator.lower_concentration_limit(
                    mass=m,
                    molecular_weight=params['molecular_weight'],
                    t_boiling=params['t_boiling'],
                    lower_concentration=params['lower_concentration']
                )
                r_lclp_mass.append(result.r_lclp)
                r_flash_mass.append(result.r_flash)

            # Расчет значений в зависимости от концентрации
            for c in concentrations:
                result = calculator.lower_concentration_limit(
                    mass=params['mass'],
                    molecular_weight=params['molecular_weight'],
                    t_boiling=params['t_boiling'],
                    lower_concentration=c
                )
                r_lclp_conc.append(result.r_lclp)
                r_flash_conc.append(result.r_flash)

            # График зависимости от массы
            ax1.plot(masses, r_lclp_mass, 'b-', label='Радиус НКПР')
            ax1.plot(masses, r_flash_mass, 'r--', label='Радиус пожара-вспышки')
            ax1.set_xlabel('Масса паров, кг')
            ax1.set_ylabel('Радиус, м')
            ax1.set_title('Зависимость радиусов от массы вещества')
            ax1.grid(True)
            ax1.legend()

            # График зависимости от концентрации
            ax2.plot(concentrations, r_lclp_conc, 'b-', label='Радиус НКПР')
            ax2.plot(concentrations, r_flash_conc, 'r--', label='Радиус пожара-вспышки')
            ax2.set_xlabel('Концентрация, % об.')
            ax2.set_ylabel('Радиус, м')
            ax2.set_title('Зависимость радиусов от концентрации')
            ax2.grid(True)
            ax2.legend()

            # Добавляем исходные данные
            info_text = (
                f"Исходные параметры:\n\n"
                f"Масса паров: {params['mass']:.1f} кг\n"
                f"Молекулярная масса: {params['molecular_weight']:.1f} кг/кмоль\n"
                f"Температура кипения: {params['t_boiling']:.1f} °C\n"
                f"НКПР: {params['lower_concentration']:.2f} % об."
            )

            # Добавляем текст на первый график
            self.figure.axes[0].text(0.02, 0.98, info_text,
                                     transform=self.figure.axes[0].transAxes,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     bbox=dict(boxstyle='round,pad=0.5',
                                               facecolor='white',
                                               alpha=0.8),
                                     fontsize=8)

            # Настраиваем отображение
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.output_text.setText(f"Ошибка при расчете: {str(e)}")