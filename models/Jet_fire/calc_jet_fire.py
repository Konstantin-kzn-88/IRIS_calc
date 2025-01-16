# -----------------------------------------------------------
# Класс предназначен для расчета факельного горения
#
# Приказ МЧС № 404 от 10.07.2009
# (C) 2023 Kuznetsov Konstantin, Kazan , Russian Federation
# email kuznetsovkm@yandex.ru
# -----------------------------------------------------------
import math

class Torch:

    def jetfire_size(self, consumption: float, type: int) -> tuple:
        """
        Расчет зон факельного горения для жидкостного факела

        Parametrs:
        :@param consumption расход, кг/с;
        :@param type расход, -; состояние вещества (0- газ, 1- газ СУГ, 2- СУГ)

        Return:
        :@return tuple(Lf, Df)
        """
        # Проверки
        if 0 in (consumption,):
            raise ValueError(f'Фукнция не может принимать нулевые параметры')

        TYPE_COEF = (12.5, 13.5, 15) # п.28, ф.П3.71-П3.72
        Lf = int(TYPE_COEF[type] * math.pow(consumption, 0.4))
        Df = math.ceil(0.15 * Lf)

        return (Lf, Df)



if __name__ == '__main__':
    pass
