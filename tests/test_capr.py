import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Prompt_regulization import CAPRClient, CAPRValidationError  # noqa: E402
from inject_capr_prompts import (  # noqa: E402
    decode_text_prompts,
    inject_prompts,
    load_label_names,
)

META_PROMPT = ROOT / "Meta-Prompt.txt"

CANONICAL_T1C = (
    "Surrounding non-enhancing FLAIR hyperintensity in head post-contrast "
    "T1-weighted MR[SEP]Resection cavity in head post-contrast T1-weighted MR"
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake CAPR client ran out of responses")
        return _FakeResponse(self.responses.pop(0))


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


def make_client(responses, *, validation_retries=0):
    return CAPRClient(
        client=_FakeOpenAI(responses),
        api_style="openai",
        model="fake-qwen3-vl",
        meta_prompt_path=META_PROMPT,
        validation_retries=validation_retries,
    )


class CAPRClientTests(unittest.TestCase):
    def test_maps_canonical_prompts_to_label_ids(self):
        client = make_client([CANONICAL_T1C])
        result = client.regularize(
            {
                2: "Surrounding non-enhancing FLAIR hyperintensit",
                4: "Resection cavity",
            }
        )
        self.assertEqual(
            result.prompts[2],
            "Surrounding non-enhancing FLAIR hyperintensity in head post-contrast "
            "T1-weighted MR",
        )
        self.assertEqual(
            result.prompts[4],
            "Resection cavity in head post-contrast T1-weighted MR",
        )
        self.assertEqual(result.attempts, 1)

    def test_accepts_lowercase_separator_fences_and_numbering(self):
        client = make_client(
            [
                "```text\n1. Spleen in abdomen CT[sep]"
                "2. Liver in abdomen CT.\n```"
            ]
        )
        result = client.regularize({1: "Spleen", 2: "Liver"})
        self.assertEqual(result.prompts[1], "Spleen in abdomen CT")
        self.assertEqual(result.prompts[2], "Liver in abdomen CT")

    def test_retries_when_count_is_wrong(self):
        client = make_client(
            [
                "Spleen in abdomen CT",
                "Spleen in abdomen CT[SEP]Liver in abdomen CT",
            ],
            validation_retries=2,
        )
        result = client.regularize({1: "Spleen", 2: "Liver"})
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.prompts[2], "Liver in abdomen CT")

    def test_rejects_the_observed_corrupted_capr_outputs(self):
        # These are the actual strings found in prompt-wrong-correct_batch_1.
        client = make_client(
            ["No-nothing tumor core.[SEP]Restriction cytotoxic."],
            validation_retries=0,
        )
        with self.assertRaises(CAPRValidationError):
            client.regularize({1: "Non-enhancing tumor core", 4: "Resection cavity"})

    def test_normalizes_known_target_spellings(self):
        client = make_client(
            [
                "gall bladder in abdomen CT[SEP]"
                "arota in abdomen CT[SEP]postcava in abdomen CT"
            ]
        )
        result = client.regularize({4: "gall bladder", 8: "arota", 9: "postcava"})
        self.assertEqual(result.prompts[4], "gallbladder in abdomen CT")
        self.assertEqual(result.prompts[8], "aorta in abdomen CT")
        self.assertEqual(result.prompts[9], "inferior vena cava in abdomen CT")

    def test_rejects_non_canonical_paraphrase_prefixes(self):
        client = make_client(
            [
                "Segmentation of liver in abdomen CT[SEP]"
                "Visualization of spleen in abdomen CT"
            ],
            validation_retries=0,
        )
        with self.assertRaises(CAPRValidationError):
            client.regularize({1: "Liver", 2: "Spleen"})

    def test_context_hint_is_sent_to_the_model(self):
        fake = _FakeOpenAI([CANONICAL_T1C])
        client = CAPRClient(
            client=fake,
            api_style="openai",
            model="fake",
            meta_prompt_path=META_PROMPT,
            validation_retries=0,
        )
        client.regularize(
            {2: "Surrounding non-enhancing FLAIR hyperintensit", 4: "Resection cavity"},
            context_hint="head post-contrast T1-weighted MR",
        )
        user_message = fake.chat.completions.calls[0]["messages"][1]["content"]
        self.assertIn("head post-contrast T1-weighted MR", user_message)


class AnthropicBackendTests(unittest.TestCase):
    def _client(self, handler):
        return CAPRClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            api_style="anthropic",
            api_key="test-token",
            base_url="https://example.test/step_plan",
            model="step-5-preview[1m]",
            meta_prompt_path=META_PROMPT,
            validation_retries=0,
        )

    def test_anthropic_messages_response_is_parsed(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["headers"] = dict(request.headers)
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": CANONICAL_T1C}],
                    "stop_reason": "end_turn",
                },
            )

        client = self._client(handler)
        result = client.regularize(
            {2: "Surrounding non-enhancing FLAIR hyperintensit", 4: "Resection cavity"}
        )
        self.assertEqual(
            result.prompts[4],
            "Resection cavity in head post-contrast T1-weighted MR",
        )
        self.assertTrue(seen["url"].endswith("/step_plan/v1/messages"))
        self.assertEqual(seen["body"]["model"], "step-5-preview[1m]")
        self.assertIn("system", seen["body"])
        self.assertEqual(seen["headers"].get("x-api-key"), "test-token")

    def test_anthropic_http_error_is_reported(self):
        def handler(request):
            return httpx.Response(
                403,
                json={
                    "error": {
                        "message": "Identity verification is required",
                        "type": "real_name_required",
                    }
                },
            )

        client = self._client(handler)
        with self.assertRaises(RuntimeError) as context:
            client.regularize({2: "Surrounding hyperintensit"})
        self.assertIn("real_name_required", str(context.exception))

    def test_max_tokens_env_override(self):
        with patch.dict(os.environ, {"CAPR_MAX_TOKENS": "1234"}, clear=False):
            client = CAPRClient(
                client=object(),
                api_style="openai",
                meta_prompt_path=META_PROMPT,
                validation_retries=0,
            )
        self.assertEqual(client.max_tokens, 1234)

    def test_stepfun_openai_env_detection(self):
        env = {
            "STEPFUN_API_KEY": "test-token",
            "STEPFUN_BASE_URL": "https://api.stepfun.com/v1",
            "STEPFUN_MODEL": "step-5-preview[1m]",
        }
        with patch.dict(os.environ, env, clear=True):
            client = CAPRClient(
                client=object(),
                meta_prompt_path=META_PROMPT,
                validation_retries=0,
            )
        self.assertEqual(client.api_style, "openai")
        self.assertEqual(client.base_url, "https://api.stepfun.com/v1")
        self.assertEqual(client.model, "step-5-preview[1m]")

    def test_anthropic_env_auto_detection(self):
        env = {
            "ANTHROPIC_AUTH_TOKEN": "test-token",
            "ANTHROPIC_BASE_URL": "https://example.test/step_plan",
            "ANTHROPIC_MODEL": "step-5-preview[1m]",
        }
        with patch.dict(os.environ, env, clear=True):
            client = CAPRClient(
                client=object(),
                meta_prompt_path=META_PROMPT,
                validation_retries=0,
            )
        self.assertEqual(client.api_style, "anthropic")
        self.assertEqual(client.base_url, "https://example.test/step_plan")
        self.assertEqual(client.model, "step-5-preview[1m]")


class InjectCAPRTests(unittest.TestCase):
    def _write_case(self, path, labels, prompts):
        gts = np.zeros((3, 3, 3), dtype=np.uint8)
        for index, label_id in enumerate(labels):
            gts[index, 0, 0] = label_id
        np.savez_compressed(
            path,
            imgs=np.full((3, 3, 3), 7, dtype=np.uint8),
            gts=gts,
            text_prompts={str(k): v for k, v in prompts.items()}
            | {"instance_label": 0},
        )

    def test_injects_canonical_prompts_by_label_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            self._write_case(
                input_dir / "case1.npz",
                [2, 4],
                {2: "Surrounding non-enhancing FLAIR hyperintensit", 4: "Resection cavity"},
            )
            self._write_case(
                input_dir / "case2.npz",
                [2],
                {2: "Surrounding non-enhancing FLAIR hyperintensit"},
            )

            client = make_client([CANONICAL_T1C])
            provenance = inject_prompts(input_dir, output_dir, capr_client=client)

            self.assertEqual(provenance["num_cases"], 2)
            self.assertTrue((output_dir / "capr_provenance.json").exists())

            with np.load(output_dir / "case1.npz", allow_pickle=True) as npz:
                self.assertEqual(npz["imgs"].shape, (3, 3, 3))
                prompts = decode_text_prompts(npz["text_prompts"])
            self.assertEqual(
                prompts,
                {
                    "2": "Surrounding non-enhancing FLAIR hyperintensity in head "
                    "post-contrast T1-weighted MR",
                    "4": "Resection cavity in head post-contrast T1-weighted MR",
                    "instance_label": 0,
                },
            )

            with np.load(output_dir / "case2.npz", allow_pickle=True) as npz:
                prompts = decode_text_prompts(npz["text_prompts"])
            self.assertEqual(set(prompts), {"2", "instance_label"})
            self.assertNotIn("4", prompts)

    def test_refuses_to_drop_a_ground_truth_label_without_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            self._write_case(
                input_dir / "case1.npz",
                [2, 4],
                {2: "Surrounding non-enhancing FLAIR hyperintensit"},
            )
            client = make_client([CANONICAL_T1C])
            with self.assertRaises(ValueError):
                inject_prompts(input_dir, output_dir, capr_client=client)

    def test_refuses_all_unknown_prompts_without_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            self._write_case(
                input_dir / "case1.npz",
                [2],
                {2: "Spleen in Unknown Unknown"},
            )
            client = make_client(["Spleen in abdomen CT"])
            with self.assertRaises(ValueError):
                inject_prompts(input_dir, output_dir, capr_client=client)

    def test_store_json_output_is_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            input_dir.mkdir()
            self._write_case(
                input_dir / "case1.npz",
                [2, 4],
                {2: "Surrounding non-enhancing FLAIR hyperintensit", 4: "Resection cavity"},
            )
            client = make_client([CANONICAL_T1C])
            inject_prompts(
                input_dir,
                output_dir,
                capr_client=client,
                store_json=True,
            )
            with np.load(output_dir / "case1.npz", allow_pickle=True) as npz:
                raw = npz["text_prompts"]
                self.assertIsInstance(raw.item(), str)
                prompts = json.loads(raw.item())
            self.assertEqual(set(prompts), {"2", "4", "instance_label"})


class LabelNameTests(unittest.TestCase):
    def test_supports_nnunet_name_to_id_and_id_to_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name_to_id = root / "name_to_id.json"
            name_to_id.write_text(
                json.dumps(
                    {
                        "labels": {
                            "background": "0",
                            "spleen": "1",
                            "right kidney": "2",
                        }
                    }
                ),
                encoding="utf-8",
            )
            id_to_name = root / "id_to_name.json"
            id_to_name.write_text(
                json.dumps(
                    {
                        "labels": {
                            "0": "background",
                            "1": "spleen",
                            "2": "right kidney",
                        }
                    }
                ),
                encoding="utf-8",
            )
            expected = {1: "spleen", 2: "right kidney"}
            self.assertEqual(load_label_names(name_to_id), expected)
            self.assertEqual(load_label_names(id_to_name), expected)


if __name__ == "__main__":
    unittest.main()
