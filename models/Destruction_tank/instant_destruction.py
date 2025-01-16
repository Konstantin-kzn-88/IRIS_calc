import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple


class LiquidSpreadModel:
    def __init__(self,
                 R: float,  # ширина резервуара, м
                 h0: float,  # начальная высота столба жидкости, м
                 a: float,  # высота обвалования, м
                 b: float,  # расстояние до обвалования, м
                 dx: float = 0.1,  # шаг по пространству
                 g: float = 9.81  # ускорение свободного падения, м/с²
                 ):
        self.R = R
        self.h0 = h0
        self.a = a
        self.b = b
        self.dx = dx
        self.g = g

        # Создаем сетку по координате x
        self.x = np.arange(0, 1.5 * b, dx)
        self.nx = len(self.x)

        # Находим индекс обвалования
        self.barrier_idx = int(b / dx)

    def solve(self, T: float, dt: float = 0.0005) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Решение системы уравнений"""
        t = np.arange(0, T, dt)
        nt = len(t)

        h = np.zeros((nt, self.nx))
        u = np.zeros((nt, self.nx))
        overflow_history = np.zeros(nt)

        # Начальные условия
        h[0, :self.barrier_idx] = np.where(self.x[:self.barrier_idx] <= self.R, self.h0, 0.0)
        u[0] = 0

        # Коэффициент искусственной вязкости
        visc = 0.05 * self.dx

        for n in range(nt - 1):
            h_curr = h[n].copy()
            u_curr = u[n].copy()

            for i in range(1, self.nx - 1):
                # Направленные разности
                if u_curr[i] > 0:
                    dh_dx = (h_curr[i] - h_curr[i - 1]) / self.dx
                    du_dx = (u_curr[i] - u_curr[i - 1]) / self.dx
                else:
                    dh_dx = (h_curr[i + 1] - h_curr[i]) / self.dx
                    du_dx = (u_curr[i + 1] - u_curr[i]) / self.dx

                dp_dx = self.g * (h_curr[i + 1] - h_curr[i - 1]) / (2 * self.dx)

                d2h_dx2 = (h_curr[i + 1] - 2 * h_curr[i] + h_curr[i - 1]) / self.dx ** 2
                d2u_dx2 = (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]) / self.dx ** 2

                h[n + 1, i] = h_curr[i] - dt * (u_curr[i] * dh_dx + h_curr[i] * du_dx) + dt * visc * d2h_dx2
                u[n + 1, i] = u_curr[i] - dt * (u_curr[i] * du_dx + dp_dx) + dt * visc * d2u_dx2

                # Обработка области вокруг обвалования
                if self.barrier_idx - 20 <= i <= self.barrier_idx:
                    # Находим максимальную высоту в расширенной области
                    region_start = max(0, self.barrier_idx - 20)
                    region_end = min(self.barrier_idx + 5, self.nx)
                    height_at_barrier = np.max(h_curr[region_start:region_end])

                    if height_at_barrier > self.a:
                        height_above = height_at_barrier - self.a

                        # Рассчитываем процент перелива
                        initial_volume = self.h0 * self.R
                        affected_width = 2.0  # характерная ширина волны
                        overflow_volume = height_above * affected_width
                        overflow_percent = (overflow_volume / initial_volume) * 100

                        # Обновляем значение перелива
                        overflow_history[n + 1] = overflow_percent

                        # print(f"\nАнализ перелива в момент t = {t[n]:.2f} с:")
                        # print(f"Высота волны: {height_at_barrier:.2f} м")
                        # print(f"Высота над обвалованием: {height_above:.2f} м")
                        # print(f"Процент перелива: {overflow_percent:.2f}%")

            # Граничные условия
            h[n + 1, 0] = h[n + 1, 1]
            u[n + 1, 0] = 0
            h[n + 1, -1] = h[n + 1, -2]
            u[n + 1, -1] = u[n + 1, -2]

            h[n + 1] = np.maximum(h[n + 1], 0)

            # Проверка условия Куранта
            max_velocity = np.max(np.abs(u[n + 1]))
            courant = max_velocity * dt / self.dx
            if courant > 0.5:
                print(f"Warning: CFL = {courant:.3f} at t = {t[n]:.3f}")

        return t, h, u, overflow_history

    def visualize(self, t: np.ndarray, h: np.ndarray, overflow_history: np.ndarray):
        """Визуализация результатов"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Ищем максимальную высоту во всем массиве
        global_max_height = np.max(h)
        global_max_time = t[np.unravel_index(np.argmax(h), h.shape)[0]]

        # График высоты жидкости
        times_to_plot = np.linspace(0, len(t) - 1, 4).astype(int)
        max_barrier_height = 0

        print("\nАнализ высот в разные моменты времени:")
        for i in times_to_plot:
            # Строим график для текущего момента времени
            ax.plot(self.x, h[i], label=f't = {t[i]:.2f} с')

            # Анализируем высоты около обвалования
            region = slice(max(0, self.barrier_idx - 20), self.barrier_idx + 1)
            local_max = np.max(h[i, region])
            max_barrier_height = max(max_barrier_height, local_max)
            local_max_idx = region.start + np.argmax(h[i, region])

            print(f"\nВремя t = {t[i]:.2f} с:")
            print(f"Максимальная высота у обвалования: {local_max:.2f} м")
            print(f"Расстояние до максимума: {self.x[local_max_idx]:.2f} м")

        # Добавляем обвалование
        barrier_x = self.x[self.barrier_idx]
        ax.vlines(barrier_x, 0, self.a, colors='r', linestyles='--', label='Обвалование')
        ax.hlines(self.a, barrier_x - 0.2, barrier_x + 0.2, colors='r', linestyles='--')

        # Добавляем текстовую информацию
        info_text = (f'Глобальный максимум: {global_max_height:.2f} м\n'
                     f'Максимум у обвалования: {max_barrier_height:.2f} м\n'
                     f'Время максимума: {global_max_time:.2f} с\n'
                     f'Максимальный перелив: {np.max(overflow_history):.2f}%')

        ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Настройка графика
        ax.set_ylim(-0.1, max(self.h0 * 1.2, self.a * 1.5))
        ax.set_xlabel('Расстояние (м)')
        ax.set_ylabel('Высота жидкости (м)')
        ax.set_title('Профиль высоты жидкости')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig('liquid_spread_results.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # Параметры задачи
    R = 5.0  # ширина резервуара
    h0 = 6.0  # начальная высота жидкости
    a = 1.0  # высота обвалования
    b = 10.0  # расстояние до обвалования

    # Создаем модель
    model = LiquidSpreadModel(R=R, h0=h0, a=a, b=b, dx=0.1)

    # Решаем уравнения
    T = 5.0  # время моделирования
    t, h, u, overflow_history = model.solve(T)

    # Визуализируем результаты
    model.visualize(t, h, overflow_history)
    print("Результаты сохранены в файл 'liquid_spread_results.png'")