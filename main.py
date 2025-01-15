# main.py
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from widgets.evaporation_widget import EvaporationCalculatorWidget
from widgets.fireball_widget import FireballCalculatorWidget
from widgets.lpg_widget import LPGCalculatorWidget
from widgets.strait_fire_widget import StraitFireCalculatorWidget
from widgets.scattering_widget import ScatteringCalculatorWidget  # Добавляем импорт
from widgets.base_calculator import BaseCalculatorTab


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Расчет аварийных ситуаций")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()

    def setup_ui(self):
        """Настройка интерфейса главного окна"""
        # Создаем вкладки
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Добавляем вкладки для разных методик
        self.tab_widget.addTab(
            BaseCalculatorTab(EvaporationCalculatorWidget),
            "Испарение жидкости"
        )
        self.tab_widget.addTab(
            BaseCalculatorTab(FireballCalculatorWidget),
            "Огненный шар"
        )
        self.tab_widget.addTab(
            BaseCalculatorTab(LPGCalculatorWidget),
            "Испарение СУГ"
        )
        self.tab_widget.addTab(
            BaseCalculatorTab(StraitFireCalculatorWidget),
            "Пожар пролива"
        )
        self.tab_widget.addTab(  # Добавляем новую вкладку
            BaseCalculatorTab(ScatteringCalculatorWidget),
            "Разлет осколков"
        )


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())