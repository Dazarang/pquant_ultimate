# Research Log

## Iter 0: Baseline
XGBClassifier on all 1,336 stocks, all 220 features, threshold=0.5.
New eval system: excess return vs market, MAE path risk, multi-horizon gate, regime breakdown.
Baseline score: **-4.5912**

Key observations:
- 71,968 signals (24% of val set) -- too many, threshold needs tuning
- Excess return ~0% -- no alpha over market
- Win rate 49.7% -- coin flip
- Knife rate 19.9% -- buying into continued drops
- Bear regime slightly better than bull (+0.30% vs -0.56%)
