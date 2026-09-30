# Identifying a wrong description

**Status:** computed. Not in the current paper.

**Result.** The number of examples after which the posterior weight on a
wrong description falls below $1/2$ (median over prompts; 95th percentile in
`results/rq2-reliability/tipping.csv`):

| $r$ | $q = 0.9$ | $q = 1 - 10^{-6}$ |
| --- | --- | --- |
| 0.5 | 5 ($d = 16$), 5 ($d = 64$) | not within 80 ($d = 16$); 23 ($d = 64$) |
| 0.05 | 1 | 2 |
| 0.001 | 1 | 1 |

A precise wrong description is abandoned after one to three examples at any
assumed reliability; only coarse ones take longer.

**Reading.** This is the mechanism behind the reversal in
[horizon-reliability.md](horizon-reliability.md). It is also an order of
magnitude faster than the crossover points reported for language models
([llms.md](llms.md)).
