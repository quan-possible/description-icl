"""make_prompts.py: the irrelevant arm of the matched stated-reliability design. ../stated-fallback/prompts.json is
the relevant arm: the c = 36 description with a stated 10% or 50% chance of being wrong and the irrelevant case
stated, n = 1, 2, 4, first 200 tasks. Here the same prompts carry each task's independently drawn value
(`invalid10`) instead, so the description is in fact irrelevant. Scoring weights the two arms by the stated chance
(exact.py, score.py). Writes prompts.json with the relevant arm's indexing: index = cell * 200 + task,
cells = [(10%, 1), (10%, 2), (10%, 4), (50%, 1), (50%, 2), (50%, 4)]."""
import json, sys
sys.path.insert(0, "..")
import llm_test as T

T.S = 2000
T.WRONG = ("There is a {q}% chance that this statement is wrong, in which case the mean was instead drawn uniformly at "
           "random from the whole numbers 100 to 999.")
t = T.make_tasks(T.MAIN)
CELLS = [(p, n) for p in (0.9, 0.5) for n in (1, 2, 4)]
rows = [dict(cell=c, task=i, p=p, n=n, prompt=T.prompt(t, (T.MAIN, 10, False, p), i, n)) for c, (p, n) in enumerate(CELLS) for i in range(200)]
json.dump(rows, open("prompts.json", "w"), indent=0)
rel = json.load(open("../stated-fallback/prompts.json"))
assert all(a["prompt"].split("\n")[0] == b["prompt"].split("\n")[0] and a["prompt"].split("\n")[2:] == b["prompt"].split("\n")[2:] for a, b in zip(rows, rel)), "arms differ beyond the stated value"
print(len(rows), "prompts"); print(rows[5 * 200 + 7]["prompt"]); print("--- relevant arm, same task"); print(rel[5 * 200 + 7]["prompt"])
