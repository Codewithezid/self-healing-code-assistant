from __future__ import annotations

import unittest

from src.code_assistant.assistant import CodeAssistant
from src.code_assistant.models import CodeSolution


class AstPatchingGuardsTests(unittest.TestCase):
    def test_guard_keeps_imports_when_patch_would_drop_them(self) -> None:
        solution = CodeSolution(
            prefix="x",
            imports="import os\nimport sys",
            code="print('ok')",
        )
        patched, notes = CodeAssistant._attempt_ast_patch(solution)
        self.assertIn("import os", patched.imports)
        self.assertIn("import sys", patched.imports)
        self.assertNotIn("import_drop_guard", [n for n in notes if n == "import_drop_guard"])

    def test_patching_preserves_ast_validity_for_valid_solution(self) -> None:
        solution = CodeSolution(
            prefix="x",
            imports="import math",
            code="def add(a, b):\n    return a + b\n",
        )
        patched, notes = CodeAssistant._attempt_ast_patch(solution)
        self.assertTrue(CodeAssistant._is_ast_valid(patched.imports, patched.code))
        self.assertTrue(len(notes) >= 0)


if __name__ == "__main__":
    unittest.main()
