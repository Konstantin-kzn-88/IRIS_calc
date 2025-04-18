# Математическая модель испарения сжиженного углеводородного газа из пролива

## Введение

Представленная математическая модель описывает процесс испарения сжиженного углеводородного газа (СУГ) при аварийном проливе на подстилающую поверхность. Модель основана на фундаментальных уравнениях теплообмена и фазовых переходов [1, 2].

## 1. Физические процессы

При разгерметизации оборудования с СУГ происходят следующие основные процессы [3]:

1. Мгновенное испарение части жидкости (flash-эффект)
2. Растекание оставшейся жидкости по поверхности
3. Интенсивное испарение за счет теплопритока от подстилающей поверхности
4. Охлаждение подстилающей поверхности

## 2. Мгновенное испарение

### 2.1 Доля мгновенного испарения

Доля мгновенного испарения определяется уравнением [4]:

φ = c_p(T_0 - T_b)/L

где:
- φ - доля мгновенного испарения (безразмерная)
- c_p - удельная теплоемкость жидкости [Дж/(кг·К)]
- T_0 - начальная температура жидкости [К]
- T_b - температура кипения при атмосферном давлении [К]
- L - удельная теплота парообразования [Дж/кг]

### 2.2 Масса мгновенного испарения

m_flash = m_0 · φ

где:
- m_flash - масса мгновенно испарившейся жидкости [кг]
- m_0 - начальная масса пролитой жидкости [кг]

## 3. Испарение с поверхности пролива

### 3.1 Тепловой поток от подстилающей поверхности

Основным механизмом теплопередачи является нестационарная теплопроводность от подстилающей поверхности [5]. Тепловой поток описывается уравнением:

q = λ_s · S · (T_s - T_b)/√(π·a_s·t)

где:
- q - тепловой поток [Вт]
- λ_s - коэффициент теплопроводности поверхности [Вт/(м·К)]
- S - площадь пролива [м²]
- T_s - начальная температура поверхности [К]
- a_s - коэффициент температуропроводности поверхности [м²/с]
- t - время с момента пролива [с]

### 3.2 Интенсивность испарения

Удельная интенсивность испарения определяется уравнением [6]:

W = q/(L·S)

где:
- W - удельная интенсивность испарения [кг/(м²·с)]

### 3.3 Масса испарившейся жидкости

Масса испарившейся жидкости в момент времени t:

m(t) = m_flash + ∫_0^t W·S·dt

## 4. Ограничения модели

1. Не учитывается конвективный теплообмен с воздухом
2. Принимается постоянная температура кипения
3. Не учитывается влияние ветра
4. Предполагается равномерное растекание жидкости

## 5. Поправочные коэффициенты

Для учета реальных условий в модель могут быть введены поправочные коэффициенты [7]:

1. K_w - коэффициент влияния ветра (1.0-2.5)
2. K_r - коэффициент шероховатости поверхности (0.6-1.0)
3. K_s - коэффициент состояния поверхности (0.7-1.0)

## Список литературы

1. Лыков А.В. Теория теплопроводности. – М.: Высшая школа, 1967. – 600 с.

2. TNO. Methods for the calculation of physical effects (Yellow Book). – The Hague, 2005.

3. ГОСТ Р 12.3.047-2012. Пожарная безопасность технологических процессов.

4. Методика определения расчетных величин пожарного риска на производственных объектах. – М.: МЧС России, 2009.

5. Хазов Г.А. Теплофизические характеристики криогенных жидкостей при аварийных проливах. – М.: МГТУ, 2015.

6. Guidelines for Chemical Process Quantitative Risk Analysis, 2nd Edition. – Center for Chemical Process Safety, AIChE, 2000.

7. Marshall V.C. Major Chemical Hazards. – Ellis Horwood Ltd., 1987.

## Приложение А: Теплофизические свойства типовых СУГ

### Пропан (C₃H₈)
- Молекулярная масса: 44.1 кг/кмоль
- Температура кипения: -42.1°C
- Теплота парообразования: 428 кДж/кг
- Теплоемкость жидкости: 2.5 кДж/(кг·К)

### Бутан (C₄H₁₀)
- Молекулярная масса: 58.1 кг/кмоль
- Температура кипения: -0.5°C
- Теплота парообразования: 385 кДж/кг
- Теплоемкость жидкости: 2.4 кДж/(кг·К)

## Приложение Б: Теплофизические свойства типовых подстилающих поверхностей

### Бетон
- Теплопроводность: 1.28 Вт/(м·К)
- Плотность: 2200 кг/м³
- Теплоемкость: 840 Дж/(кг·К)

### Грунт (суглинок)
- Теплопроводность: 0.8 Вт/(м·К)
- Плотность: 1600 кг/м³
- Теплоемкость: 960 Дж/(кг·К)