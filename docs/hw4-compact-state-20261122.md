# Компактная память прошлых исходов поверх победившего GAM

План до результатов: 22 ноября 14:34:09 UTC,
SHA `89a3a5968778afcea7bd6389b723e08ce9a134b10263806cb2d0e5781c8e5449`.
Исходный E-transform сохранён без изменения: его design prefix EXACT.
Добавлены четыре численных поля history_fail_rate/recent365_fail_rate/
history_n_visits/last_routine_age_days и три OHE-категории lag1_result/
last_routine_result/lag1_inspection_type. Counts/возраст — fixed log1p;
fit-only median/clip/scale и четыре missing flags. Никакие 176 code-индикаторов
из широкой истории не добавлялись.

Вторая заранее выбранная форма добавляет только существующий spline базиса
lastRoutineLoad × OHE lastRoutineResult, максимум20 произведений, с fit-only
масштабированием. Автоматических всех попарных взаимодействий нет.
Две формы × C=.003/.01/.03/.1 — восемь inner-рецептов. Каждый outer использует
исключительно AP предыдущего inner-квартала; финал — majority/mean-inner tie.

| Модель | Средний AP | Δ к matched E | Неотрицательных кварталов |
|---|---:|---:|---:|
| State raw алгоритм | .277531548 | −.000919050 | 1/3 |
| 75%context+25%state | .270660213 | −.000678415 к такому же E-blend | 0/3 |

Дополнение немного улучшило Q4 raw, но ухудшило Q3 и Q1. Оба gate не пройдены.
Fixed final count был бы обычный state/C=.01; полный fit и новые CSV не делали.
Это не доказывает общую неинформативность прошлых исходов или причину неудачи
старой широкой модели: в этой ограниченной форме они не победили matched E.

Аудит PASS: 25 checkpoint переиграны EXACT, 21 внешний AP, выбор только по
предыдущему inner и strict resume без fit. Новых Q2/Q3 метрик и uploads нет.
Аудит: `analysis/hw4-six-hour-20261122/additive/compact-state/audit/verification.json`.
