from __future__ import annotations

import unittest
from pathlib import Path


class CodingArenaFrontendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.index_html = (self.project_root / "public" / "index.html").read_text(encoding="utf-8")
        self.app_js = (self.project_root / "public" / "app.js").read_text(encoding="utf-8")
        self.styles_css = (self.project_root / "public" / "styles.css").read_text(encoding="utf-8")

    def test_index_contains_arena_button_and_dual_panels(self) -> None:
        self.assertIn('id="arenaToggleBtn"', self.index_html)
        self.assertIn('id="arenaWrap"', self.index_html)
        self.assertIn('id="arenaMsgsRag"', self.index_html)
        self.assertIn('id="arenaMsgsNormal"', self.index_html)
        self.assertIn("Coding Arena: RAG vs Normal", self.index_html)

    def test_app_runs_parallel_rag_and_normal_requests(self) -> None:
        self.assertIn("function toggleArenaMode()", self.app_js)
        self.assertIn("Promise.all([", self.app_js)
        self.assertIn("const ragPayload = {", self.app_js)
        self.assertIn('runtime_profile: "goated"', self.app_js)
        self.assertIn("rag_enabled: true", self.app_js)
        self.assertIn("const normalPayload = {", self.app_js)
        self.assertIn("rag_enabled: false", self.app_js)

    def test_styles_define_arena_layout_and_button(self) -> None:
        self.assertIn(".arena-btn{", self.styles_css)
        self.assertIn(".arena-panels{", self.styles_css)
        self.assertIn("grid-template-columns:minmax(0,1fr) minmax(0,1fr);", self.styles_css)
        self.assertIn(".arena-panel{", self.styles_css)

    def test_index_contains_attachment_controls(self) -> None:
        self.assertIn('id="attachmentInput"', self.index_html)
        self.assertIn('id="attachBtn"', self.index_html)
        self.assertIn('id="attachMode"', self.index_html)


if __name__ == "__main__":
    unittest.main()
