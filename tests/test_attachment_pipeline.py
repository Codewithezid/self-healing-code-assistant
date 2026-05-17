from __future__ import annotations

import unittest
import zipfile
from io import BytesIO

from src.code_assistant.web import (
    _compose_prompt_with_attachments,
    _compose_prompt_with_attachment_refs,
    _extract_docx_text,
)


class AttachmentPipelineTests(unittest.TestCase):
    def test_compose_prompt_with_attachments(self) -> None:
        prompt = "Fix this bug"
        attachments = [
            {"filename": "main.py", "content": "def add(a,b):\n    return a+b"},
            {"filename": "notes.txt", "content": "edge case: empty list"},
        ]
        combined = _compose_prompt_with_attachments(prompt, attachments)
        self.assertIn("Fix this bug", combined)
        self.assertIn("[main.py]", combined)
        self.assertIn("edge case: empty list", combined)
        self.assertIn("Attachment Context", combined)

    def test_extract_docx_text_from_minimal_zip(self) -> None:
        content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Hello</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>World</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        payload = BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr("word/document.xml", content)
        text = _extract_docx_text(payload.getvalue())
        self.assertIn("Hello", text)
        self.assertIn("World", text)

    def test_compose_prompt_with_attachment_refs(self) -> None:
        prompt = "Refactor API"
        refs = [{"filename": "api_design.pdf", "content": "ignored here"}]
        combined = _compose_prompt_with_attachment_refs(prompt, refs)
        self.assertIn("Refactor API", combined)
        self.assertIn("api_design.pdf", combined)
        self.assertIn("indexed in RAG", combined)


if __name__ == "__main__":
    unittest.main()
