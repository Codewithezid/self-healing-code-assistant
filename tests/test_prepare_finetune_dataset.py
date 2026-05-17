from __future__ import annotations

import unittest

from scripts.prepare_finetune_dataset import normalize_examples, parse_dataset_triple, row_to_messages


class PrepareFineTuneDatasetTests(unittest.TestCase):
    def test_parse_dataset_triple(self) -> None:
        dataset, config, split = parse_dataset_triple("owner/ds:default:train")
        self.assertEqual(dataset, "owner/ds")
        self.assertEqual(config, "default")
        self.assertEqual(split, "train")

    def test_row_to_messages_handles_instruction_style(self) -> None:
        row = {"instruction": "Write add fn", "input": "in python", "output": "def add(a,b): return a+b"}
        mapped = row_to_messages(row)
        self.assertIsNotNone(mapped)
        self.assertEqual(mapped["messages"][0]["role"], "user")
        self.assertEqual(mapped["messages"][1]["role"], "assistant")

    def test_normalize_examples_dedupes_and_filters(self) -> None:
        rows = [
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "x" * 100}]},
            {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "x" * 100}]},
            {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "tiny"}]},
        ]
        normalized = normalize_examples(rows, min_assistant_chars=20, max_assistant_chars=200)
        self.assertEqual(len(normalized), 1)


if __name__ == "__main__":
    unittest.main()
