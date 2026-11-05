"""make_prompts.py: the paper's stated-reliability condition (the c = 36 description with a stated 10% or 50% chance of
being wrong, n = 1, 2, 4, first 200 tasks of the main run) with one change: the warning now states what a wrong
description implies, the uniform over 100..999 that the Bayes-optimal reference (llm_test.ideal_mean) already assumes.
Writes prompts.json; index = cell * 200 + task, cells = [(10%, 1), (10%, 2), (10%, 4), (50%, 1), (50%, 2), (50%, 4)]."""
import json, sys
sys.path.insert(0, "..")
import llm_test as T

T.S = 2000
T.WRONG = ("There is a {q}% chance that this statement is wrong, in which case the mean was instead drawn uniformly at "
           "random from the whole numbers 100 to 999.")
t = T.make_tasks(T.MAIN)
CELLS = [(p, n) for p in (0.9, 0.5) for n in (1, 2, 4)]
rows = [dict(cell=c, task=i, p=p, n=n, prompt=T.prompt(t, (T.MAIN, 10, True, p), i, n)) for c, (p, n) in enumerate(CELLS) for i in range(200)]
json.dump(rows, open("prompts.json", "w"), indent=0)
print(len(rows), "prompts"); print(rows[3 * 200 + 5]["prompt"])
