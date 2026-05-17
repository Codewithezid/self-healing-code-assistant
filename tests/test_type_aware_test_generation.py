from __future__ import annotations

from src.code_assistant.assistant import CodeAssistant
from src.code_assistant.models import CodeSolution


def test_generate_unit_tests_uses_type_aware_arguments() -> None:
    solution = CodeSolution(
        prefix="binary search",
        imports="",
        code=(
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    if not isinstance(arr, list):\n"
            "        raise TypeError('arr must be list')\n"
            "    return -1\n"
        ),
    )
    tests = CodeAssistant._generate_unit_tests(solution)
    assert "binary_search([1, 2, 3], 2)" in tests
    assert "binary_search([], 0)" in tests
    assert "binary_search('not-a-list', 'bad')" in tests
    assert "Signature constraints" in tests


def test_generate_unit_tests_skips_methods_with_self() -> None:
    solution = CodeSolution(
        prefix="class method",
        imports="",
        code=(
            "class A:\n"
            "    def foo(self, x: int) -> int:\n"
            "        return x\n"
        ),
    )
    tests = CodeAssistant._generate_unit_tests(solution)
    assert tests == ""
