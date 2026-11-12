# Evaluation — what is measured, and what the numbers do not mean

## 1. Two splits, enforced in code

| file | items | used for |
|---|---|---|
| `datasets/dev.jsonl` | 70 (7 per category) | every candidate evaluation while searching |
| `datasets/heldout.jsonl` | 40 (4 per category) | only the final report |

`load_splits()` fails if any id **or prompt** appears in both files, and the loop
calls `assert_no_heldout_leak()` on every evaluation, so a held-out item reaching
the search is a hard error rather than a silent bias. Regenerate both with
`python scripts/build_datasets.py` (the generator asserts disjointness too).

## 2. Ten categories

1. `roman_hinglish` — Roman Hindi with spelling variation and informal typing
2. `devanagari_hindi` — Devanagari input, Devanagari answer expected
3. `mixed_code_switch` — Hindi+English in one sentence ("Mera laptop slow chal raha hai")
4. `context_followup` — "ye / usko / isse / iska / wahi" reference resolution
5. `instruction_following` — bullets, word limits, "only a number", language constraints
6. `question_answering` — arithmetic, hardware, programming, general facts
7. `natural_response` — small talk, empathy, identity; must not lecture or dodge
8. `spelling_robustness` — `smjh nhi aya`, `ye kya kr rha hai`, ALL CAPS, no punctuation
9. `technical_hindi` — Docker, Git, REST, weights, cron, SSL in Hindi/Hinglish
10. `factuality` — stable facts + fabrication bans

Each item carries a rubric: `must_include` (alternative token groups, all must
appear), `answer_tokens` (any one proves a correct answer), `expect_script`,
`max_words` / `min_words` / `expect_bullets`, `must_not_include`, `fact_banned`.

## 3. Seven dimensions

| dimension | how it is computed | weight |
|---|---|---|
| `intent` | do the semantic token groups appear | 0.25 |
| `correctness` | verifiable answer token present (null when the item has none) | 0.20 |
| `script` | Devanagari ratio for Devanagari requests; Latin ratio for Roman requests | 0.15 |
| `compliance` | length limits, bullet count, banned phrases | 0.15 |
| `relevance` | non-empty, no dodge markers, no unrequested "Here is the translation" | 0.10 |
| `naturalness` | repeated 4-gram loop penalty, length sanity | 0.10 |
| `factual` | fabrication bans (null when the item has none) | 0.05 |

`composite` = weighted mean over the *applicable* dimensions; an empty output scores
0 outright. `null` means "not applicable to this item" and is excluded from the mean
rather than counted as zero.

## 4. Accept / reject

```json
"acceptance": { "min_composite_gain": 0.005, "max_category_regression": 0.03, "min_abs_score": 0.0 }
```

A candidate is kept only if **all** hold: composite gain ≥ `min_composite_gain`,
no category worse than `-max_category_regression`, and composite ≥ `min_abs_score`.
Otherwise it is rejected, its delta status becomes `rejected`, and the weights are
rebuilt from base + kept deltas. Rejections are recorded, not hidden — the report
shows both counts.

## 5. Honest limitations

* **Heuristic, not human.** A rubric cannot judge wit, correctness of an open
  explanation, or whether a Hindi sentence sounds native. It can catch: wrong
  script, missing intent, dodged requests, unrequested translation, broken length
  limits, degenerate loops, fabricated banned claims.
* **Token gaps.** A correct answer worded differently from the rubric's token groups
  can lose `intent` points; that is a false negative, and it makes the scorer
  conservative (it under-rewards good edits rather than over-rewarding bad ones).
* **Small sets.** 70 dev / 40 held-out items across 10 categories is ~1 item per
  category per point of composite; differences below ~0.03 are noise. `min_composite_gain`
  is set accordingly and `composite_stdev` is reported alongside every mean.
* **Optional LLM judge.** `--judge` adds an NVIDIA-scored `naturalness` field when a
  key is available; it never overrides the deterministic accept/reject decision, so a
  flaky judge cannot silently rewrite the verdict.
* **No claim of "solved".** A composite gain on this rubric means "measurably better
  on these probes", nothing more.

## 6. Reproducing

```bash
python -m aladdin.cli baseline --split dev          # writes logs/baseline_dev/{results.json,summary.csv}
python -m aladdin.cli evaluate --split heldout --experiment <id> --out heldout.json
```

`results.json` holds per-item dims, notes, output and the full generation protocol
(`max_new_tokens`, `temperature`, `top_k`, `seed`), so two runs with the same config
are comparable line by line.
