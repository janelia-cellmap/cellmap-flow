"""Tests for finetuning CLI parsing."""

import unittest

from cellmap_flow.finetune.finetune_cli import build_arg_parser


class FinetuneCliParserTests(unittest.TestCase):
    def test_parser_accepts_script_model_type(self):
        parser = build_arg_parser()

        args = parser.parse_args(
            [
                "--model-type",
                "script",
                "--model-script",
                "/tmp/model.py",
                "--corrections",
                "/tmp/corrections",
                "--output-dir",
                "/tmp/output",
            ]
        )

        self.assertEqual(args.model_type, "script")
        self.assertEqual(args.model_script, "/tmp/model.py")


if __name__ == "__main__":
    unittest.main()
