#!/usr/bin/env python3
"""Optional small pilot metric script for the 6-function summarization table.

This is not the repository-level experiment; it reproduces the lightweight
summarization metric calculation used as a toy/pilot baseline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


@dataclass
class Example:
    name: str
    reference: str
    name_only: str
    rule_based: str
    zero_shot: str


DATA = [
    Example("factorial", "Calculate the factorial of a non-negative integer n.", "Factorial.", "Calculate the factorial of a non-negative integer.", "Compute the factorial of a non-negative integer."),
    Example("mean", "Compute the arithmetic mean of a list of numbers.", "Mean.", "Compute the arithmetic mean of a list of numbers.", "Return the average value of a numeric list."),
    Example("is_prime", "Determine if a number is prime.", "Prime check.", "Determine if a number is prime.", "Check whether an integer is a prime number."),
    Example("sort_numbers", "Sort a list of numbers in ascending order.", "Sort numbers.", "Sort a list of numbers in ascending order.", "Return the input numbers sorted from smallest to largest."),
    Example("fibonacci", "Generate the nth Fibonacci number.", "Fibonacci.", "Generate the nth Fibonacci number.", "Compute the Fibonacci value at a given index."),
    Example("find_max", "Find the maximum element in a list.", "Find maximum.", "Find the maximum element in a list.", "Return the largest value from the list."),
]


def bleu1(ref: str, hyp: str) -> float:
    rt = ref.lower().split()
    ht = hyp.lower().split()
    rc = Counter(rt)
    overlap = sum(min(rc[t], ht.count(t)) for t in set(ht))
    return overlap / max(1, len(ht))


def avg(vals):
    return sum(vals) / len(vals)


def main() -> None:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    systems = ["name_only", "rule_based", "zero_shot"]
    print("system,bleu1,rougeL,meteor")
    for sys_name in systems:
        b, r, m = [], [], []
        for ex in DATA:
            hyp = getattr(ex, sys_name)
            b.append(bleu1(ex.reference, hyp))
            r.append(scorer.score(ex.reference, hyp)["rougeL"].fmeasure)
            # METEOR token input; WordNet synonym matching may be unavailable offline.
            m.append(meteor_score([ex.reference.split()], hyp.split()))
        print(f"{sys_name},{avg(b):.3f},{avg(r):.3f},{avg(m):.3f}")


if __name__ == "__main__":
    main()
