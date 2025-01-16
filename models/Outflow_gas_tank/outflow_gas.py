import numpy as np
import matplotlib

matplotlib.use('Agg')  # Установка бэкенда до импорта pyplot
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class GasFlowModel:
    def __init__(self, V, T0_celsius, P0_MPa, gamma, R, D_mm, Pa_MPa=0.101325):
        """
        Инициализация модели истечения газа

        Параметры:
        V: объем резервуара (м³)
        T0_celsius: начальная температура газа (°C)
        P0_MPa: начальное давление газа (МПа)
        gamma: показатель адиабаты газа
        R: газовая постоянная (Дж/(кг·К))
        D_mm: диаметр отверстия (мм)
        Pa_MPa: атмосферное давление (МПа)
        """
        self.V = V
        self.T0 = T0_celsius + 273.15  # перевод в Кельвины
        self.P0 = P0_MPa * 1e6  # перевод в Па
        self.gamma = gamma
        self.R = R
        self.A = np.pi * (D_mm / 2000) ** 2  # перевод диаметра в площадь в м²
        self.Pa = Pa_MPa * 1e6  # перевод в Па

        # Начальная масса газа
        self.m0 = (self.P0 * V) / (R * self.T0)

    def mass_flow_rate(self, P, T):
        """
        Расчет массового расхода газа через отверстие
        """
        beta = self.Pa / P  # Отношение давлений

        if beta > (2 / (self.gamma + 1)) ** (self.gamma / (self.gamma - 1)):  # Докритический режим
            mdot = self.A * P * np.sqrt(2 * self.gamma / (self.R * T) * \
                                        (beta ** (2 / self.gamma) - beta ** ((self.gamma + 1) / self.gamma)) / (
                                                    self.gamma - 1))
        else:  # Критический режим
            mdot = self.A * P * np.sqrt(self.gamma / (self.R * T)) * \
                   ((2 / (self.gamma + 1)) ** ((self.gamma + 1) / (2 * (self.gamma - 1))))

        return mdot

    def derivatives(self, state, t):
        """
        Система дифференциальных уравнений
        state[0] - масса газа
        state[1] - давление
        state[2] - температура
        """
        m, P, T = state

        # Проверяем условия прекращения истечения
        if P <= self.Pa * 1.01:  # Давление близко к атмосферному (запас 1%)
            return [0, 0, 0]

        if m <= 0.1 or T <= 200:  # Физические ограничения
            return [0, 0, 0]

        mdot = self.mass_flow_rate(P, T)

        # Изменение массы
        dmdt = -mdot

        # Относительное изменение массы
        relative_mass_change = dmdt / m

        # Изменение температуры (из уравнения адиабаты)
        dTdt = T * (self.gamma - 1) * relative_mass_change

        # Изменение давления (из уравнения состояния идеального газа)
        dPdt = P * (relative_mass_change + dTdt / T)

        return [dmdt, dPdt, dTdt]

    def simulate(self, t_span, dt):
        """
        Моделирование процесса истечения

        Параметры:
        t_span: время моделирования (с)
        dt: шаг по времени (с)
        """
        t = np.arange(0, t_span, dt)
        initial_state = [self.m0, self.P0, self.T0]

        solution = odeint(self.derivatives, initial_state, t)

        return t, solution


# Пример использования
if __name__ == "__main__":
    # Параметры для воздуха
    V = 100.0  # м³
    T0 = 20.0  # °C
    P0 = 1.0  # МПа
    gamma = 1.4  # показатель адиабаты для воздуха
    R = 287.05  # Дж/(кг·К)
    D = 30.0  # мм (диаметр отверстия)

    # Создание модели
    model = GasFlowModel(V, T0, P0, gamma, R, D)

    # Моделирование
    t_span = 800  # секунд
    dt = 0.01
    t, solution = model.simulate(t_span, dt)

    # Настройка стиля
    plt.style.use('default')

    # Создание фигуры с заданным размером
    fig = plt.figure(figsize=(15, 10), dpi=100)

    # Добавление общего заголовка с исходными данными
    fig.suptitle('Моделирование истечения газа из резервуара\n' +
                 f'Объем: {V} м³, Давление: {P0} МПа, Температура: {T0}°C\n' +
                 f'Диаметр отверстия: {D} мм, Параметры газа (γ={gamma}, R={R} Дж/(кг·К))',
                 fontsize=14, y=0.98)

    # График давления
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(t, solution[:, 1] / 1e6, linewidth=2, color='#1f77b4')
    ax1.set_title('Давление в резервуаре', fontsize=12, pad=10)
    ax1.set_xlabel('Время (с)', fontsize=10)
    ax1.set_ylabel('Давление (МПа)', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # График массы
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, solution[:, 0], linewidth=2, color='#2ca02c')
    ax2.set_title('Масса газа в резервуаре', fontsize=12, pad=10)
    ax2.set_xlabel('Время (с)', fontsize=10)
    ax2.set_ylabel('Масса (кг)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # График температуры
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, solution[:, 2] - 273.15, linewidth=2, color='#ff7f0e')
    ax3.set_title('Температура газа', fontsize=12, pad=10)
    ax3.set_xlabel('Время (с)', fontsize=10)
    ax3.set_ylabel('Температура (°C)', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # График массового расхода
    mdot = np.array([model.mass_flow_rate(P, T) for P, T in zip(solution[:, 1], solution[:, 2])])
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, mdot, linewidth=2, color='#d62728')
    ax4.set_title('Массовый расход', fontsize=12, pad=10)
    ax4.set_xlabel('Время (с)', fontsize=10)
    ax4.set_ylabel('Расход (кг/с)', fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))

    plt.tight_layout()

    # Сохранение графика в файл
    plt.savefig('gas_outflow_results.png', bbox_inches='tight')
    plt.close()  # Закрытие фигуры для освобождения памяти