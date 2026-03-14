from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_assistant.assistant import CodeAssistant, CodeSolution


BENCHMARK_CASES = [
    {
        "name": "merge_intervals",
        "prompt": (
            "Write a Python function merge_intervals(intervals) that accepts a list of "
            "[start, end] pairs, merges overlapping or touching intervals, sorts unsorted input, "
            "and returns the merged list. Include a short example call."
        ),
        "tests": """
def normalize_intervals(items):
    return [list(item) for item in items]

assert normalize_intervals(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]])) == [[1, 6], [8, 10], [15, 18]]
assert normalize_intervals(merge_intervals([[5, 7], [1, 2], [2, 4], [4, 5]])) == [[1, 7]]
assert normalize_intervals(merge_intervals([])) == []
""",
    },
    {
        "name": "topological_sort",
        "prompt": (
            "Write a Python function topological_sort(graph) that takes a dictionary mapping each "
            "node to a list of neighbors and returns a valid topological ordering. If the graph "
            "contains a cycle, raise ValueError. Include a short example call."
        ),
        "tests": """
order = topological_sort({
    'cook': ['eat'],
    'shop': ['cook'],
    'code': [],
    'eat': [],
})
assert set(order) == {'cook', 'shop', 'code', 'eat'}
assert order.index('shop') < order.index('cook') < order.index('eat')
try:
    topological_sort({'a': ['b'], 'b': ['a']})
    raise AssertionError('Expected ValueError for cyclic graph')
except ValueError:
    pass
""",
    },
    {
        "name": "lru_cache",
        "prompt": (
            "Write a Python class LRUCache with methods get(key) and put(key, value). It should "
            "evict the least recently used item when capacity is exceeded and return -1 for missing "
            "keys. Include a short example call."
        ),
        "tests": """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1
cache.put(3, 3)
assert cache.get(2) == -1
cache.put(4, 4)
assert cache.get(1) == -1
assert cache.get(3) == 3
assert cache.get(4) == 4
""",
    },
    {
        "name": "csv_parser",
        "prompt": (
            "Write a Python function parse_csv_line(line) that parses a single CSV line and returns "
            "a list of values. It must handle commas inside double quotes and escaped double quotes "
            "inside quoted fields. Include a short example call."
        ),
        "tests": r'''
assert parse_csv_line('a,b,c') == ['a', 'b', 'c']
assert parse_csv_line('"a,b",c') == ['a,b', 'c']
assert parse_csv_line('"he said ""hi""",x') == ['he said "hi"', 'x']
''',
    },
    {
        "name": "roman_to_int",
        "prompt": (
            "Write a Python function roman_to_int(s) that converts a Roman numeral to an integer. "
            "Support standard subtractive notation such as IV, IX, XL, XC, CD, and CM. Raise "
            "ValueError for an empty string. Include a short example call."
        ),
        "tests": """
assert roman_to_int('III') == 3
assert roman_to_int('MCMXCIV') == 1994
assert roman_to_int('XLII') == 42
try:
    roman_to_int('')
    raise AssertionError('Expected ValueError for empty input')
except ValueError:
    pass
""",
    },
    {
        "name": "expression_evaluator",
        "prompt": (
            "Write a Python function evaluate_expression(expr) that evaluates an arithmetic "
            "expression string containing +, -, *, /, parentheses, and spaces. Do not use eval. "
            "Division should produce floats when needed. Include a short example call."
        ),
        "tests": """
assert evaluate_expression('2 + 3 * 4') == 14
assert evaluate_expression('(2 + 3) * 4') == 20
assert abs(evaluate_expression('18 / (3 * 2) + 1') - 4.0) < 1e-9
assert evaluate_expression('7 - 10 + 5') == 2
""",
    },
    {
        "name": "bst_validation",
        "prompt": (
            "Write a Python class TreeNode and a function is_valid_bst(root) that returns True if a "
            "binary tree is a valid binary search tree and False otherwise. Include a short example "
            "call."
        ),
        "tests": """
root = TreeNode(2, TreeNode(1), TreeNode(3))
assert is_valid_bst(root) is True
bad = TreeNode(5, TreeNode(1), TreeNode(4, TreeNode(3), TreeNode(6)))
assert is_valid_bst(bad) is False
""",
    },
    {
        "name": "group_anagrams",
        "prompt": (
            "Write a Python function group_anagrams(words) that groups words that are anagrams of "
            "each other. Return a list of groups, where each group is sorted alphabetically and the "
            "overall list is sorted by the first word in each group. Include a short example call."
        ),
        "tests": """
result = group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat'])
assert result == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
assert group_anagrams([]) == []
""",
    },
]


def run_case(assistant: CodeAssistant, case: dict[str, str]) -> tuple[bool, str]:
    result = assistant.run(case["prompt"])
    solution = result.get("generation")
    if not isinstance(solution, CodeSolution) or result.get("error") != "no":
        return False, "assistant did not produce a validated solution"

    snippet = "\n\n".join(
        part
        for part in [solution.imports.strip(), solution.code.strip(), case["tests"].strip()]
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if completed.returncode == 0:
        return True, f"passed in {result.get('iterations')} iteration(s)"
    error_text = completed.stderr.strip() or completed.stdout.strip() or "semantic test failed"
    return False, error_text


def main() -> int:
    assistant = CodeAssistant(max_iterations=3, validation_timeout_seconds=5)
    passed = 0

    print("Complex benchmark results")
    print("-------------------------")
    for case in BENCHMARK_CASES:
        try:
            ok, detail = run_case(assistant, case)
        except Exception as exc:
            ok, detail = False, str(exc)
        passed += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['name']}: {detail}")

    total = len(BENCHMARK_CASES)
    print()
    print(f"Measured complex benchmark pass rate: {passed}/{total} = {passed / total:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
