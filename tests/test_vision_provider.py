import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from deploy import l4filter


class VisionProviderTests(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("vision_encoder.onnx")
        self.options = l4filter._session_options(1)
        session_patch = patch.object(l4filter, "_VisionSession")
        providers_patch = patch.object(l4filter.ort, "get_available_providers", return_value=[l4filter.CPU_PROVIDER])
        preload_patch = patch.object(l4filter.ort, "preload_dlls", create=True)
        self.session = session_patch.start()
        self.providers = providers_patch.start()
        self.preload = preload_patch.start()
        self.addCleanup(session_patch.stop)
        self.addCleanup(providers_patch.stop)
        self.addCleanup(preload_patch.stop)

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
        self.preload.assert_not_called()

    def test_cuda_taken_from_managed_runner_environment(self):
        active = [l4filter.CUDA_PROVIDER, l4filter.CPU_PROVIDER]
        self.providers.return_value = active
        expected_session = self.session.return_value
        expected_session.get_providers.return_value = active

        def open_session(*args, **kwargs):
            self.preload.assert_called_once_with(directory="")
            return expected_session

        self.session.side_effect = open_session
        with patch.dict(os.environ, {l4filter.VISION_PROVIDER_ENV: "cuda"}):
            selected = l4filter._create_vision_session(self.model_path, self.options, None)
        self.assertIs(selected, expected_session)
        self.session.assert_called_once_with(
            str(self.model_path), sess_options=self.options,
            providers=[(l4filter.CUDA_PROVIDER, {"use_tf32": "0"}), l4filter.CPU_PROVIDER],
            enable_fallback=False,
        )
        selected.disable_fallback.assert_called_once_with()

    def test_cuda_without_preload_api_uses_system_libraries(self):
        active = [l4filter.CUDA_PROVIDER, l4filter.CPU_PROVIDER]
        self.providers.return_value = active
        self.session.return_value.get_providers.return_value = active
        with patch.object(l4filter.ort, "preload_dlls", None):
            selected = l4filter._create_vision_session(self.model_path, self.options, "cuda")
        self.assertIs(selected, self.session.return_value)
        self.preload.assert_not_called()

    def test_unavailable_cuda_fails_before_starting_session(self):
        with self.assertRaisesRegex(l4filter.VisionProviderError, "CUDAExecutionProvider is unavailable"):
            l4filter._create_vision_session(self.model_path, self.options, "cuda")
        self.session.assert_not_called()
        self.preload.assert_not_called()

    def test_session_fallback_is_not_reported_as_cuda(self):
        self.providers.return_value = [l4filter.CUDA_PROVIDER, l4filter.CPU_PROVIDER]
        self.session.return_value.get_providers.return_value = [l4filter.CPU_PROVIDER]
        with self.assertRaisesRegex(l4filter.VisionProviderError, "could not activate CUDAExecutionProvider"):
            l4filter._create_vision_session(self.model_path, self.options, "cuda")


class WebGpuProviderTests(unittest.TestCase):
    def setUp(self):
        self.model_path = Path("vision_encoder.onnx")
        self.options = MagicMock()
        self.plugin = SimpleNamespace(
            get_ep_name=lambda: l4filter.WEBGPU_PROVIDER, get_library_path=lambda: "/plugins/webgpu.so",
        )
        self.device = SimpleNamespace(ep_name=l4filter.WEBGPU_PROVIDER)
        patches = {
            "session": patch.object(l4filter, "_VisionSession"),
            "providers": patch.object(l4filter.ort, "get_available_providers", return_value=[l4filter.CPU_PROVIDER]),
            "register": patch.object(l4filter.ort, "register_execution_provider_library", create=True),
            "devices": patch.object(
                l4filter.ort, "get_ep_devices", create=True,
                return_value=[SimpleNamespace(ep_name=l4filter.CPU_PROVIDER), self.device],
            ),
            "registered": patch.object(l4filter, "_registered_ep_libraries", set()),
            "module": patch.dict(sys.modules, {"onnxruntime_ep_webgpu": self.plugin}),
        }
        for name, item in patches.items():
            setattr(self, name, item.start())
            self.addCleanup(item.stop)
        self.session.return_value.get_providers.return_value = [l4filter.WEBGPU_PROVIDER, l4filter.CPU_PROVIDER]

    def test_plugin_registered_once_and_bound_to_gpu_device(self):
        for _ in range(2):
            selected = l4filter._create_vision_session(self.model_path, self.options, "webgpu")
        self.register.assert_called_once_with(l4filter.WEBGPU_PROVIDER, "/plugins/webgpu.so")
        self.options.add_provider_for_devices.assert_called_with([self.device], {})
        self.session.assert_called_with(str(self.model_path), sess_options=self.options)
        selected.disable_fallback.assert_called_with()

    def test_builtin_webgpu_provider_skips_plugin(self):
        self.providers.return_value = [l4filter.WEBGPU_PROVIDER, l4filter.CPU_PROVIDER]
        l4filter._create_vision_session(self.model_path, self.options, "webgpu")
        self.register.assert_not_called()
        self.session.assert_called_once_with(
            str(self.model_path), sess_options=self.options,
            providers=[(l4filter.WEBGPU_PROVIDER, {}), l4filter.CPU_PROVIDER], enable_fallback=False,
        )

    def test_missing_plugin_package(self):
        with patch.dict(sys.modules, {"onnxruntime_ep_webgpu": None}):
            with self.assertRaisesRegex(l4filter.VisionProviderError, "requirements-webgpu.txt"):
                l4filter._create_vision_session(self.model_path, self.options, "webgpu")
        self.session.assert_not_called()

    def test_no_gpu_adapter(self):
        self.devices.return_value = [SimpleNamespace(ep_name=l4filter.CPU_PROVIDER)]
        with self.assertRaisesRegex(l4filter.VisionProviderError, "no compatible GPU adapter"):
            l4filter._create_vision_session(self.model_path, self.options, "webgpu")
        self.session.assert_not_called()

    def test_cpu_fallback_is_not_reported_as_webgpu(self):
        self.session.return_value.get_providers.return_value = [l4filter.CPU_PROVIDER]
        with self.assertRaisesRegex(l4filter.VisionProviderError, "could not activate WebGpuExecutionProvider"):
            l4filter._create_vision_session(self.model_path, self.options, "webgpu")


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
