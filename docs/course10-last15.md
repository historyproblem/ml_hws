# Последние15минут: три проверенных файла

Пользователь сообщил15минут до своего ограничения на попытки, попросил
быструю проверку или использование оставшихся сегодня отправок. Уточнил,
что основной сезонныйForest ещё не отправлен. Последний известный
остаток3; новых upload агентом0. Более ранний сохранённый23:30дедлайн
не подменяет это актуальное указание и не считается проверкой liveKaggle.

На готовых признаках проверены четыре заранее заданных изменения:
max_features=sqrt,depth8,leaf40,half-life180. Июнь–август используются
для раннего отбора,Sep–Oct только для прежнего сезонного контроля и
раннихtop2новых. Контроли переиспользованы. Новый поиск не улучшил
основной сезонныйForest:

| Кандидат | Pooled Sep–Oct |
|---|---:|
| **Основной F0+season** |**2.645327**|
| F2+season, предыдущий проверенный кандидат |2.646235|
| F0+season,sqrt |2.646853|
| F0+season,leaf40 |2.647517|

Дляsqrt разница с основным+0.001526,95%customer-cluster bootstrap
CI[−0.003104,+0.005985]; дляleaf40+0.002190,CI[−0.000896,+0.005316].
Ни один не доказал улучшение. Основной остаётся первым выбором.

Дополнительно полностью рефитнута/воспроизведена обычная одиночнаяMLP42,
но matchingCPUпроверка далаSep2.628408,Oct2.720333,pooled2.675985.
Это хуже сезонногоForest на0.030658,CI[+0.010310,+0.050392]. Её отправка
не рекомендуется. Проверка не подменяется прежнимиCUDAметриками.

Полные рефиты50852официальных строк выполнены дляF2+season иF0+season/sqrt.
Оба standalone-ноутбука заново построили признаки из официальных CSV,
проверили50852таргета с нулевой ошибкой, выполнилиSep/Oct и финальныйfit.
CSV3188Id/sampleorder/finite/nonnegative/SHA/ZIPCRC прошли проверку.
Максимальное отличие от ноутбука3.18e-12дляF2,5.68e-13дляsqrt.
Прежний основнойF0+season уже проверен аналогично. Только обычные
RandomForestсеминара9,15/49базовых агрегатов+21прошлогодний/календарный,
D0,directlogMSE,seed42; прогнозы моделей не смешиваются.

Уникальные имена готовыхCSV в порядке приоритета:

1. `outputs/course10-last15-ready/submission_01_rf_season.csv`
2. `outputs/course10-last15-ready/submission_02_rf_baskets_season.csv`
3. `outputs/course10-last15-ready/submission_03_rf_season_sqrt.csv`

CSV01SHA911d8869...,02SHA6442e94b...,03SHA75c81447....
Полные SHA/точные метрики: `outputs/course10-last15-ready/priority.json`.
Каждая копия побайтно совпадает с соответствующим проверенным рефитом.
02/03 — исследовательские альтернативы для использования оставшихся
попыток, локально немного хуже01. Улучшение public/private не обещается;
набор не выбирался под leaderboard-множители или тестовые ответы.

Ноутбуки и полныеZIP:
`outputs/course10-{season-forest,baskets-season,season-sqrt}-delivery/`.
MLPсохранена отдельно как отвергнутый контроль. Журналы,matchingCPUпрогнозы,
bootstrap и доказательства выполнения — `analysis/course10-last15-20261115/`.

Воспроизведение последней проверки и выбранногоsqrt:

```bash
python scripts/course10_forest_followup.py --config configs/course10_season_last15.json --out outputs/course10-season-last15
python scripts/course10_forest_followup.py --config configs/course10_season_last15.json --out outputs/course10-season-last15 --refit rf_season_sqrt
python scripts/package_course10_forest_followup.py --profile rf_season_sqrt --run outputs/course10-season-last15 --out outputs/course10-season-sqrt-delivery
```

Новые конфиги не изменяют существующие исходники моделей. Чужие
untrackedfinal-checks сохранены. После пользовательских отправок необходимо
отдельно зафиксировать фактическиеscore/остаток попыток; сейчас они неизвестны.
