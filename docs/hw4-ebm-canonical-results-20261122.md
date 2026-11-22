# HW4: greedy и canonical-style EBM

22 ноября 2026. Отдельный согласованный прогон завершён за 163.7 с: **24 estimator-fit = 12 INNER + 9 OUTER (три seed) + 3 full**. Допущен только заранее фиксированный probability blend с прежней07. Весь EBM в этом раунде закрыт; дальнейшего перебора его параметров нет.

## Что проверено

[План до HW4-fit](hw4-ebm-canonical-plan-20261122.md) задаёт четыре рецепта compact94. A/B меняют относительно прежних controlled main/pairs300 только `greedy_ratio:0→10`. C/D проверяют ограниченный canonical-style upstream алгоритм: LR .015, bins1024/interaction64, min_leaf4, L2=0, bags8, internal random validation .15, cap2000, ES50, tolerance1e-5; main либо те же 10 заранее заданных pairs. Это не полный default upstream: bags/cap/ES и scope interactions ограничены явно.

C/D меняют согласованный набор upstream-style параметров одновременно. Поэтому положительный результат относится к этому алгоритму целиком; его нельзя причинно приписывать отдельно greedy, bags или bins. Matched A/B изолируют greedy, но не были выбраны ни в одном outer.

Все 12 внутренних fit закончены до первого outer. Для 2023Q3/Q4 выбраны canonical pairs10, для 2024Q1 — canonical main; выбор только по previous-inner AP. Агрегированный full recipe — `ebm_canonical_pairs10` (mean OLD INNER AP .29572447). Seeds17/2026 повторяют выбранный алгоритм без внешней перенастройки. Внутренний random split из прошлых FIT-строк является частью native EBM, не называется temporal validation. Внешние inner/outer строки и labels в fit не передавались.

## Итог old development, среднее трёх seed

| Вектор | Mean AP | Δ к прежней07 | Δ≥0, кварталы | Worst Δ к07 | Допуск |
| --- | ---: | ---: | ---: | ---: | --- |
| Новый GAM другой ветки | .27699741 | +.00615707 | 3/3 | +.00405323 | диагностическая опора |
| EBM raw | .27329656 | +.00245621 | 2/3 | −.00522265 | нет: worst ниже −.005 |
| `.75*07+.25*EBM` | .27201829 | +.00117794 | 2/3 | −.00041583 | да |
| Прежняя07 | .27084035 | 0 | 3/3 | 0 | reference |
| Context04 | .26750641 | −.00333393 | 0/3 | −.00588998 | diagnostic |

Для raw квартальные Δ к07: −.00522265 / +.00503274 / +.00755856; для blend: −.00041583 / +.00267843 / +.00127123. Site-bootstrap95% CI для mean blend Δ: [−.00107810; +.00335989]. Это nominal CI после многократного использования old dev; независимое подтверждение gain не заявляется.

Новый GAM raw выше EBM-blend во всех трёх кварталах. EBM raw немного выше GAM в Q4 (+.00039650), поэтому заранее зафиксированный **BOTH**-domination guard не запретил full. Такой допуск не делает EBM более сильным финалистом, чем GAM; он сохраняет потенциально отличающийся сигнал для общего зафиксированного shortlist.

## Артефакты и аудит

Output: `analysis/hw4-six-hour-20261122/ebm-canonical/`. `confirmed-oof.csv` содержит official metadata, `ebm_raw` (raw mean3), `ebm_blend75`, `previous07`, `context04`, `new_gam_raw`. Raw full-test mean3 восстанавливается из `models/ebm_canonical_pairs10__full__s{42,17,2026}/prediction.npy` без нового fit.

Готов `submission_ebm_canonical_blend75.csv`, 2246 строк в точном sample order; SHA256 `4f852438dc1546fa2b93a49dec062439b0e00a3721b6fccde150b19d7159f0fd`. Формула ровно `.75*previous07_test+.25*mean3_EBM_test`. Для recent diagnostic можно refit **этот фиксированный** `CANONICAL_RECIPES['ebm_canonical_pairs10']` на всех строках до оцениваемого периода; cap/ES policy/terms неизменны, внешнего count selector нет.

`audit/verification.json`: все 24 job SHA/reload flags проверены; independent loaded-model forwards совпали EXACT; все inner AP/logloss и quarter AP воспроизведены с ошибкой 0; оба OOF mean/blend реконструированы EXACT; target/ID alignment и test CSV проверены. Signature `4a82ec632dfeb92743bbd2972be6dd5616f631ad1ef542cfab3893146d7c1ea0` доказывает зафиксированную provenance и согласованность артефактов, не является независимой записью исполнения до сохранения модели.

Runtime изолирован, основные пакеты не обновлялись. 7 synthetic backend/runner QA PASS; native SYNTHETIC probe C/D на 1000 строках выполнял все реальные bags/cap/ES до HW4. Новых внешних Q2/Q3-метрик и Kaggle uploads: **0**. Классическая ветка шестичасового раунда: **141 estimator-fit = 81 boosting + 9 TD + 27 cyclic EBM + 24 canonical EBM**, без старых108 и synthetic QA.

Источник алгоритма: [upstream estimator 0.7.8](https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_ebm.py), [greedy training loop](https://github.com/interpretml/interpret/blob/v0.7.8/python/interpret-core/interpret/glassbox/_ebm/_boost.py).
