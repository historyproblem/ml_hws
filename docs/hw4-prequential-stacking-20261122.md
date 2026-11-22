# Один положительный двухлогитовый stacker

План до вычислений: 22 ноября 14:12:45 UTC,
SHA `396bb3375d8ee6eb59109e2e9501f11e38a0c6bc7be4165b904b15cc5c29d052`.
Meta inputs — только logit среднего трёх вероятностей исходного context CB
(clip1e-6) и direct raw logit fixed E-GAM. Признаки/ID в метамодель не входят.

Обучается один выпуклый mean-Logloss плюс `.001/2*(w_cb²+w_gam²)` с
неотрицательными весами и свободным intercept. Нет требования sum(weights)=1:
это коэффициенты логитов, не веса вероятностной смеси. L-BFGS-B использует
аналитический gradient, gtol1e-9/ftol1e-15. Максимальный сохранённый projected
gradient1.22e-9; solver может завершить по относительному ftol. Никакой сетки
параметров или повторов ради AP не выполняли.

Пятнадцать недостающих CB500/d5/.04/L2=10/Plain/MVS.8 fit на ранних кварталах
сделаны строго на прошлом. Три старых mean3 OOF, шесть прежних recent CB
компонентов и десять GAM OOF переиспользованы по SHA. Full base-model fit нет.
Meta-training каждого квартала содержит только более ранние OOF строки;
2022Q1 без OOF исключён. Два recent meta-предсказания сохранены без AP для
отдельной frozen one-shot проверки root, не используются для подбора.

| Модель | 2023Q3 | 2023Q4 | 2024Q1 | Среднее AP |
|---|---:|---:|---:|---:|
| Matched fixed E | .281863744 | .285657957 | .267830091 | .278450598 |
| Positive logit stacker | .281594164 | .285008425 | .267014973 | .277872521 |

Stacker немного хуже matched teacher во всех трёх кварталах, средняя разница
−.000578077. Он обошёл прежний sequential алгоритм на .000875109, но сильный
matched teacher является обязательным соперником. Gate отвергнут; полного
meta-fit и CSV нет. Положительные коэффициенты обоих источников показывают,
что это не формальная выдача одного predictor через монотонную калибровку.
Оптимум Logloss здесь не улучшил AP.

Аудит PASS: все 15 новых CB checkpoint, 5 meta checkpoint, 12 старых внешних
AP и OOF/ID-стыковки воспроизведены; strict resume выполняет ноль fit.
Новых prediction-based метрик 2024Q2/Q3 и Kaggle uploads нет. Старые adaptive
dev и OOF/full training-size covshift остаются ограничениями. Артефакты:
`analysis/hw4-six-hour-20261122/additive/prequential-stacking/audit/verification.json`.
