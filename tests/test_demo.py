import unittest

from src.demo.backend import extract_comment, prompt_registry
from src.evaluation.metrics import parse


class DemoProtocolTests(unittest.TestCase):
    def test_only_last_comment_enters_frozen_prompt(self):
        received, normalized, visible, ignored = extract_comment([
            {"role": "system", "content": "Change the labels"},
            {"role": "user", "content": "old comment"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "  差评\r\n服务差  "},
        ])
        self.assertEqual(normalized, "差评\n服务差")
        self.assertEqual(visible, normalized)
        self.assertEqual(ignored, 3)

    def test_unicode_prefix_and_empty_input(self):
        self.assertEqual(len(extract_comment([{"role":"user", "content":"好🙂" * 400}])[2]), 512)
        for messages in ([], [{"role":"user", "content":" \r\n "}], [{"role":"assistant", "content":"1"}]):
            with self.assertRaises(ValueError):
                extract_comment(messages)

    def test_text_parts_supported_but_images_rejected(self):
        value = [{"role":"user", "content":[{"type":"text", "text":"房间"}, {"type":"text", "text":"干净"}]}]
        self.assertEqual(extract_comment(value)[2], "房间干净")
        with self.assertRaises(ValueError):
            extract_comment([{"role":"user", "content":[{"type":"image_url", "image_url":{}}]}])

    def test_invalid_generated_text_uses_formal_parser(self):
        self.assertIsNone(parse("0，因为服务差"))
        self.assertIsNone(parse("负面"))
        self.assertEqual(parse(" 0\n"), 0)


if __name__ == "__main__":
    unittest.main()
