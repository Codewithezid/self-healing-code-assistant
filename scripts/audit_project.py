from __future__ import annotations

import os
import subprocess
import sys
import warnings
from contextlib import contextmanager
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


class FakeChain:
    def __init__(self, responses: list[CodeSolution]) -> None:
        self.responses = responses
        self.index = 0

    def invoke(self, _: dict[str, object]) -> CodeSolution:
        if self.index >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self.index]
        self.index += 1
        return response


@contextmanager
def patched_build_chain(responses: list[CodeSolution]):
    original = CodeAssistant._build_chain

    def fake_build_chain(self: CodeAssistant) -> FakeChain:
        return FakeChain(responses)

    CodeAssistant._build_chain = fake_build_chain
    try:
        yield
    finally:
        CodeAssistant._build_chain = original


def run_command(args: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def check_cli_help() -> tuple[bool, str]:
    code, stdout, _ = run_command([sys.executable, "main.py", "--help"])
    ok = code == 0 and "Run the LangGraph self-correcting code assistant." in stdout
    return ok, "CLI help renders correctly" if ok else "CLI help failed"


def check_compile() -> tuple[bool, str]:
    code, _, stderr = run_command([sys.executable, "-m", "compileall", "main.py", "src"])
    ok = code == 0
    return ok, "Python sources compile successfully" if ok else f"Compile failed: {stderr}"


def check_missing_key_error() -> tuple[bool, str]:
    env = os.environ.copy()
    env["MISTRAL_API_KEY"] = ""
    completed = subprocess.run(
        [sys.executable, "main.py", "print hello"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    ok = completed.returncode == 1 and "Configuration error:" in completed.stdout
    return ok, "Missing API key fails cleanly" if ok else "Missing API key handling failed"


def check_mocked_success_first_try() -> tuple[bool, str]:
    responses = [
        CodeSolution(
            prefix="Return a hello world function.",
            imports="",
            code="def hello():\n    return 'hello world'",
        )
    ]
    with patched_build_chain(responses):
        assistant = CodeAssistant(max_iterations=3)
        result = assistant.run("Write a hello world function.")
    ok = result["error"] == "no" and result["iterations"] == 1
    return ok, "Graph succeeds on first valid attempt" if ok else "Graph first-attempt success failed"


def check_mocked_retry_then_success() -> tuple[bool, str]:
    responses = [
        CodeSolution(
            prefix="Broken import example.",
            imports="import not_a_real_package",
            code="",
        ),
        CodeSolution(
            prefix="Corrected palindrome helper.",
            imports="",
            code="def is_palindrome(value):\n    cleaned = ''.join(ch.lower() for ch in value if ch.isalnum())\n    return cleaned == cleaned[::-1]",
        ),
    ]
    with patched_build_chain(responses):
        assistant = CodeAssistant(max_iterations=3)
        result = assistant.run("Write a palindrome checker.")
    ok = result["error"] == "no" and result["iterations"] == 2
    return ok, "Graph retries after failure and then succeeds" if ok else "Retry loop failed"


def check_mocked_max_iteration_stop() -> tuple[bool, str]:
    responses = [
        CodeSolution(
            prefix="Still broken.",
            imports="import another_missing_package",
            code="",
        )
    ]
    with patched_build_chain(responses):
        assistant = CodeAssistant(max_iterations=2)
        result = assistant.run("Keep failing.")
    ok = result["error"] == "yes" and result["iterations"] == 2
    return ok, "Graph stops at max iterations on repeated failure" if ok else "Max iteration guard failed"


def run_live_examples() -> tuple[int, int, list[str]]:
    if not os.getenv("MISTRAL_API_KEY"):
        return 0, 0, ["Live audit skipped: MISTRAL_API_KEY is not configured."]

    benchmark_cases = [
        {
            "name": "fibonacci",
            "prompt": (
                "Write a Python function fibonacci(n) that returns a list with the first n "
                "Fibonacci numbers. Include a short example call."
            ),
            "tests": (
                "assert fibonacci(1) == [0]\n"
                "assert fibonacci(5) == [0, 1, 1, 2, 3]\n"
                "assert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]\n"
            ),
        },
        {
            "name": "palindrome",
            "prompt": (
                "Write a Python function is_palindrome(s) that ignores spaces, punctuation, "
                "and capitalization. Include a short example call."
            ),
            "tests": (
                "assert is_palindrome('racecar') is True\n"
                "assert is_palindrome('A man, a plan, a canal: Panama!') is True\n"
                "assert is_palindrome('hello') is False\n"
            ),
        },
        {
            "name": "factorial",
            "prompt": (
                "Write a Python function factorial(n) for non-negative integers. Raise "
                "ValueError for negative input. Include a short example call."
            ),
            "tests": (
                "assert factorial(0) == 1\n"
                "assert factorial(5) == 120\n"
                "try:\n"
                "    factorial(-1)\n"
                "    raise AssertionError('Expected ValueError for negative input')\n"
                "except ValueError:\n"
                "    pass\n"
            ),
        },
        {
            "name": "fizzbuzz",
            "prompt": (
                "Write a Python function fizzbuzz(n) that returns a list of strings from 1 to n "
                "using Fizz, Buzz, and FizzBuzz rules. Include a short example call."
            ),
            "tests": (
                "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
                "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
            ),
        },
        {
            "name": "reverse_words",
            "prompt": (
                "Write a Python function reverse_words(text) that returns the words in reverse "
                "order separated by single spaces. Include a short example call."
            ),
            "tests": (
                "assert reverse_words('hello world') == 'world hello'\n"
                "assert reverse_words('one two three') == 'three two one'\n"
            ),
        },
        {
            "name": "count_vowels",
            "prompt": (
                "Write a Python function count_vowels(text) that returns the number of vowels in "
                "a string, case-insensitive. Include a short example call."
            ),
            "tests": (
                "assert count_vowels('hello') == 2\n"
                "assert count_vowels('AEIOU') == 5\n"
                "assert count_vowels('rhythm') == 0\n"
            ),
        },
    ]
    passed = 0
    notes: list[str] = []

    assistant = CodeAssistant(max_iterations=3)

    for case in benchmark_cases:
        try:
            result = assistant.run(case["prompt"])
            solution = result.get("generation")
            if not isinstance(solution, CodeSolution) or result.get("error") != "no":
                notes.append(f"FAIL: {case['name']} -> assistant did not produce a validated solution")
                continue

            snippet = "\n\n".join(
                part
                for part in [solution.imports.strip(), solution.code.strip(), case["tests"]]
                if part
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-c", snippet],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            success = completed.returncode == 0
            passed += int(success)
            if success:
                notes.append(f"PASS: {case['name']}")
            else:
                error_text = completed.stderr.strip() or completed.stdout.strip() or "semantic test failed"
                notes.append(f"FAIL: {case['name']} -> {error_text}")
        except Exception as exc:
            notes.append(f"ERROR: {case['name']} -> {exc}")
    return passed, len(benchmark_cases), notes


def main() -> int:
    checks = [
        check_cli_help,
        check_compile,
        check_missing_key_error,
        check_mocked_success_first_try,
        check_mocked_retry_then_success,
        check_mocked_max_iteration_stop,
    ]

    passed = 0
    total = 0

    print("Offline audit results")
    print("---------------------")
    for check in checks:
        ok, message = check()
        passed += int(ok)
        total += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")

    live_passed, live_total, live_notes = run_live_examples()
    if live_total:
        print()
        print("Live example results")
        print("--------------------")
        for note in live_notes:
            print(note)
        print(
            f"Measured live semantic benchmark pass rate: "
            f"{live_passed}/{live_total} = {live_passed / live_total:.2%}"
        )
    else:
        print()
        print("Live example results")
        print("--------------------")
        for note in live_notes:
            print(note)

    print()
    print(f"Offline pass rate: {passed}/{total} = {passed / total:.2%}")
    print()
    print("Capability note")
    print("---------------")
    if live_total:
        print(
            "The live pass rate above is only a small-sample benchmark, not the exact overall accuracy "
            "of the project."
        )
    else:
        print(
            "Exact end-to-end accuracy cannot be measured in this environment because the live Mistral "
            "API path is not configured."
        )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
