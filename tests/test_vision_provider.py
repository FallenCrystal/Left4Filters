import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from deploy import l4filter


class VisionProviderTests(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("vision_encoder.onnx")
        self.options = l4filter._session_options(1)
        session_patch = patch.object(l4filter, "_VisionSession")
        providers_patch = patch.object(l4filter.ort, "get_available_providers", return_value=[l4filter.CPU_PROVIDER])
        self.session = session_patch.start()
        self.providers = providers_patch.start()
        self.addCleanup(session_patch.stop)
        self.addCleanup(providers_patch.stop)

    def test_default_and_explicit_cpu(self):
        with patch.dict(os.environ):
            os.environ.pop(l4filter.VISION_PROVIDER_ENV, None)
            l4filter._create_vision_session(self.model_path, self.options, None)
            self.session.assert_called_once_with(
                str(self.model_path), sess_options=self.options, providers=[l4filter.CPU_PROVIDER],
            )
            self.session.reset_mock()
            os.environ[l4filter.VISION_PROVIDER_ENV] = "cuda"
            l4filter._create_vision_session(self.model_path, self.options, "cpu")
            self.session.assert_called_once_with(
                str(self.model_path), sess_options=self.options, providers=[l4filter.CPU_PROVIDER],
            )

    def test_cuda_taken_from_managed_runner_environment(self):
        active = ["CUDAExecutionProvider", l4filter.CPU_PROVIDER]
        self.providers.return_value = active
        self.session.return_value.get_providers.return_value = active
        with patch.dict(os.environ, {l4filter.VISION_PROVIDER_ENV: "cuda"}):
            selected = l4filter._create_vision_session(self.model_path, self.options, None)
        self.assertIs(selected, self.session.return_value)
        self.session.assert_called_once_with(
            str(self.model_path), sess_options=self.options,
            providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), l4filter.CPU_PROVIDER],
            enable_fallback=False,
        )
        selected.disable_fallback.assert_called_once_with()

    def test_unavailable_cuda_fails_before_starting_session(self):
        with self.assertRaisesRegex(l4filter.VisionProviderError, "CUDAExecutionProvider is unavailable"):
            l4filter._create_vision_session(self.model_path, self.options, "cuda")
        self.session.assert_not_called()

    def test_session_fallback_is_not_reported_as_cuda(self):
        self.providers.return_value = ["CUDAExecutionProvider", l4filter.CPU_PROVIDER]
        self.session.return_value.get_providers.return_value = [l4filter.CPU_PROVIDER]
        with self.assertRaisesRegex(l4filter.VisionProviderError, "could not activate CUDAExecutionProvider"):
            l4filter._create_vision_session(self.model_path, self.options, "cuda")


class InferenceTimingTests(unittest.TestCase):
    def test_cold_and_cached_prediction_log(self):
        with Image.new("RGB", (256, 256), "#c0c4c8") as image:
            contents = io.BytesIO()
            image.save(contents, format="PNG")
        with self.assertLogs("left4filters.inference", "INFO") as records:
            clf = l4filter.Left4Filter(num_threads=1, vision_provider="cpu")
            first = clf.predict(title="Workshop addon", image=contents.getvalue())
            cached = clf.predict(title="Workshop addon", image=contents.getvalue())
        self.assertEqual(first, cached)
        messages = "\n".join(records.output)
        self.assertIn("Model ready provider=CPUExecutionProvider", messages)
        self.assertIn("Image preprocessing images=1 latency_ms=", messages)
        self.assertIn("Vision inference provider=CPUExecutionProvider images=1 latency_ms=", messages)
        self.assertIn("Vision feature cache hit images=1", messages)
        self.assertIn("Prediction provider=CPUExecutionProvider images=1 latency_ms=", messages)
