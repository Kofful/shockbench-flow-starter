# Записка для передачі (стан на 2026-10-07)

Для кого: Олесь на іншому комп'ютері і нова сесія Claude Code. Детальний план — [PLAN.md](../PLAN.md), усі числа —
[experiments/LOG.md](../experiments/LOG.md). Тут — коротко, де ми і що далі.

## Де ми

| Агент | Small dev (20 еп.) | Статус |
| --- | --- | --- |
| `template` (шле максимум) | 0.409 | точка відліку |
| `rl` (Владислав, гілка `origin/cuda`) | 0.528 | |
| **`mpc`** (наш) | **0.727** | готовий до завантаження: `sbf check` проходить на Small (≤ 0.18 с/тиждень) і Full (≤ 0.78 с) |
| `mpc_residual` (наш) | = `mpc`, поки без `params.json` | тренується |

Нічого ще не завантажено на Codabench.

## Що ми з'ясували (і чому не робимо деяких речей)

1. **MPC — правильна основа.** Пакет організаторів має baseline `mpc_det` (0.729). Ми перенесли його на сервер:
   `agents/mpc/` = скопійований код пакета (`sbflow/`) + `scipy.optimize.linprog` замість highspy. LP з нашого
   спостереження ідентичний LP зі справжнього на всіх тижнях (перевірено).
2. **Score 1 недосяжний.** «Ідеальний план» — LP, що спрощує правила гри (розподіл енергії, випуск із черги на протоці).
   Навіть MPC, що знає все майбутнє, має лише 0.80 (`mpc_perfect_T`).
3. **Прогноз майже нічого не дає.** Ідеальне знання майбутнього на 24 тижні: +0.057. Початок закриття проток за
   попередженнями не передбачається (AUC 0.56; тривога перед закриттям зростає лише на +0.06σ). Тривалість закриття
   передбачається добре (AUC 0.85), але навіть ідеальне її знання дає лише +0.007. **Класифікатори в агент не йдуть.**
4. **Довший горизонт MPC:** +0.008 за ×4 CPU — не беремо.
5. **Основний резерв (~0.20):** LP неточно моделює правила гри → систематичні помилки MPC. Їх атакує `mpc_residual`:
   12 поправок θ ∈ [−1, 1] за групами маршрутів (однакові на Small і Full) поверх плану MPC.
6. **Тренування `mpc_residual` триває.** Перший запуск на ноутбуці (8 ГБ RAM, swap): ~6 хв на покоління; на 0-му
   поколінні жоден варіант (σ = 0.15 і 0.10) не кращий за θ = 0. Продовжити на ПК (32 ГБ RAM).
7. **CUDA не потрібна:** MPC і симулятор працюють лише на CPU; сервер теж CPU.

## Нові файли

| Файл | Що робить |
| --- | --- |
| `agents/mpc/agent.py` | MPC-агент для сервера |
| `agents/mpc_residual/agent.py` | MPC + 12 поправок з `params.json` |
| `agents/*/sbflow/` | скопійований пакет організаторів — **не редагувати**, генерує `scripts/vendor_mpc.py` |
| `scripts/vendor_mpc.py` | копіює пакет у обидва агенти (перезапустити після оновлення shockbench-flow) |
| `examples/train_residual.py` | тренування `mpc_residual` (еволюційна стратегія; root 1001, перевірка на 9001) |
| `examples/score_baselines.py` | оцінка baseline-ів пакета і діагностичних MPC (`mpc_det@H`, `mpc_perfect*`) |
| `examples/collect_data.py` | сирі дані для класифікаторів у `data/` (не в git) |
| `notebooks/strait_classifiers.ipynb` | класифікатори проток (onset / still) |
| `examples/watch_agent.py` | вікно з повзунком по тижнях: карта, витрати за статтями, події тижня |
| `examples/episode_events.py` | що сталося в епізоді по тижнях |
| `examples/run_agent.py` | агент тиждень за тижнем у терміналі |
| `examples/08_agent_losses.py`, `09_compare_plans.py`, `src/sbf_starter/diagnostics.py` | від Владислава: графіки витрат, порівняння з ідеальним планом (`--clairvoyant`) |

## Налаштування на новому комп'ютері

```bash
git clone https://github.com/Kofful/shockbench-flow-starter && cd shockbench-flow-starter
git checkout oles
uv sync
uv add --dev scikit-learn pandas        # лише для notebook; у pyproject їх ще немає
# Windows: .venv\Scripts\activate       macOS / Linux: source .venv/bin/activate
sbf check mpc --task=small              # перевірка, що все працює
```

`data/` і кеш еталонів (`~/.cache/shockbench-flow`) у git не потрапляють: кеш порахується сам при першому запуску
(Small dev ~3 хв), дані — `python examples/collect_data.py --task=small --entropy=1001 --episodes=2000` (~20 хв).

## Що далі

1. Тренування на ПК: `python examples/train_residual.py` (15 поколінь × 12 варіантів × 32 епізоди; пише
   `agents/mpc_residual/params.json` лише якщо краще за `mpc` на root 9001).
2. Якщо `mpc_residual` кращий: `sbf compare mpc_residual mpc --task=small --entropy=9001 --episodes=64`, потім Full.
3. Ще не зроблено: `sbf evaluate mpc --task=full --entropy=9001 --episodes=16` (Full — фінальна таблиця).
4. Перше завантаження `mpc` на Codabench — рішення команди (3 спроби на день).

## Для нової сесії Claude Code

- Спілкування українською, простими словами, коротко; без зайвих коментарів у коді й notebook.
- Довгі запуски (тренування, оцінювання) Олесь запускає сам — давати команди, не запускати.
- Почати з цього файлу, `PLAN.md` і `experiments/LOG.md`.
