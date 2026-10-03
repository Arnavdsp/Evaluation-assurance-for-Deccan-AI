# Evaluation Assurance

Which of an AI evaluator's PASS verdicts should a human re-check?

> Everything here runs on synthetic data where I know the ground truth. None of it is Deccan data, and it says nothing about how Helix performs.

## The question

An automated evaluator (LLM judge, rubric scorer, verifier) sometimes marks a run as PASS when it actually failed. For example, the agent said it saved a record but the DB disagrees, or it skipped a required step. Nobody can re-check every PASS. So if reviewers only have time for X% of them, which X% should they look at?

My approach: rank the PASS verdicts using execution checks the evaluator didn't use (DB read-back, tool-argument audit, replay/ordering check, completion checklist). Then train a small model on a few hundred human-labelled PASS verdicts to learn how much each check should count.

## Results (20 seeds, ± = 95% CI)

About 25% of PASS verdicts are silently wrong in this synthetic setup.

| Method | Caught @ 20% review | Precision @ 20% | Caught @ 5% | Precision @ 5% |
|---|---|---|---|---|
| Random spot-check | 20.4% | 25% | 5.0% | 25% |
| Model on the evaluator's own inputs | 27.5% ± 1.1 | 34% | 7.4% | 37% |
| Rule: count fired checks (no ML) | 45.1% ± 0.5 | 56% | 13.4% | 66% |
| **Execution checks + logistic regression** | **48.4% ± 0.8** | **60%** | **18.5%** | **92%** |
| Execution checks + random forest | 47.1% ± 1.5 | 59% | 17.4% | 86% |
| Both feature sets + random forest | 45.9% ± 1.3 | 57% | 17.3% | 86% |

<img width="1080" height="690" alt="image" src="https://github.com/user-attachments/assets/7b1d643a-9333-4a31-b94b-555f3af137be" />


What I take from this:

- The gain comes from using evidence the evaluator didn't look at. A better model on the evaluator's own inputs barely beats random.
- A plain rule gets most of the way. The model only really pulls ahead when the budget is tight: at 5% review, 92% of what it flags is a real failure, against 66% for the rule.
- Logistic regression beat the random forest. It had about 260 labels, lower calibration error (0.042 vs 0.053), and its weights double as "why was this flagged" for the reviewer.
- At a more realistic failure rate (5% of PASS verdicts wrong), reviewing 10% still catches 43% of failures. Only about 1 in 5 flags is real, though (random: 1 in 20). Precision is what drives reviewer cost.
- Two failure types (wrong tool argument, ordering violation) are capped by how often their checks are logged at all (55% and 35%). That's a logging problem, not a modelling one.

Sanity checks: a pure-noise feature changes nothing (-0.2%). Catch drops gradually as the signals get noisier (48 → 45 → 40 → 33%). On a harder task mix it goes 48 → 42%. 150 labels already get you 45%.

## How it got here

- **v1**: the auditor only saw task metadata, basically what the judge saw. It caught ~4%.
- **v2**: AUC 0.91, looked great. It was leakage: one feature was built from the same formula as the label, and a two-line rule matched the model.
- **v3**: I split every failure into a hidden cause and a noisy observed signal, and added a leakage check. That got 47% vs 28%. A single seed said "combined > structural"; across 20 seeds it didn't hold.
- **v4**: I added random and rule baselines to v3. The rule got close, so the claim got narrower. I also switched to logistic regression and tested lower failure rates.

## Where it would sit

```
agent run -> evaluator -> PASS -> [execution checks] -> score + reasons -> top-k to human review
                                                                                  |
                                                            labels -> canary set -> weekly refit
```

It runs as a small step after the evaluator: ~15 weights, CPU, batch or streaming, no external API in the path. The same idea could flag RL rollouts where the verifier said PASS but the state checks disagree, which looks a lot like reward hacking.

`evaluation-assurance/src/jev_adapter.py` is an optional hook for TypeSafe's Jev on free-text evidence. It isn't used for any number above.

## Real-data pilot

Take 200–500 anonymized traces with existing verdicts, whatever checks are actually logged, and trusted outcomes. Swap them in for `src/evaluator_mock.py` and the `obs_*` columns, then re-run. If it doesn't beat both random and the rule at the target budget, it shouldn't ship.

## Limitations

- The failure modes and check quality are made up (plausible, not measured). That's the main thing a pilot would test.
- The lower failure rates are simulated by subsampling.
- Shift and robustness tests are synthetic-to-synthetic.
- It assumes ~300 labelled PASS verdicts are cheap enough to get.

## Run

```bash
cd evaluation-assurance
pip install -r requirements.txt
python experiments/run_all.py     # ~2 min, writes results/ and figures/
python -m pytest -q               # metrics, leakage guard, one-seed smoke run
```

`run_all.py` is seeded: re-running it reproduces the CSVs in `results/` exactly.

The notebook (`evaluation-assurance/notebooks/evaluation_assurance_demo.ipynb`) is self-contained and runs on Colab.

```
evaluation-assurance/
src/          data_generation, evaluator_mock, audit_models, metrics, pipeline, jev_adapter (optional)
experiments/  headline, realistic_prevalence, per_mode, calibration_negative_control,
              robustness_shift, canary_size, make_figures, run_all
tests/        pytest suite (runs in CI)
```
