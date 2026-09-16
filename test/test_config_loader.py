import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from duolaAgent.config.loader import load_runtime_config, provider_config


class ConfigLoaderTest(unittest.TestCase):
    def test_load_runtime_config_from_duola_json_and_env_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "duola.json"
            workspace = root / "workspace-from-file"
            config_path.write_text(
                json.dumps(
                    {
                        "agents": {
                            "defaults": {
                                "provider": "anthropic",
                                "model": "file-model",
                                "workspace": str(workspace),
                                "stream": False,
                            }
                        },
                        "providers": {
                            "anthropic": {
                                "api_key": "${TEST_DUOLA_API_KEY}",
                                "api_base": "https://example.invalid",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "TEST_DUOLA_API_KEY": "secret",
                    "DUOLA_API_KEY": "",
                    "DUOLA_API_BASE": "",
                    "ANTHROPIC_API_KEY": "",
                    "ANTHROPIC_BASE_URL": "",
                    "DUOLA_MODEL": "env-model",
                    "DUOLA_STREAM": "true",
                    "DUOLA_RETRY_ATTEMPTS": "5",
                },
                clear=False,
            ):
                config = load_runtime_config(config_path=config_path)

            defaults = config.agents.defaults
            provider = provider_config(config, defaults.provider)
            self.assertEqual(defaults.model, "env-model")
            self.assertEqual(defaults.workspace, str(workspace.resolve()))
            self.assertTrue(defaults.stream)
            self.assertEqual(defaults.retry_attempts, 5)
            self.assertEqual(provider.api_key, "secret")
            self.assertEqual(provider.api_base, "https://example.invalid")

    def test_explicit_provider_base_selects_openai_compatible_provider_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                os.environ,
                {
                    "DUOLA_MODEL": "mimo-v2.5",
                    "DUOLA_PROVIDER": "openai-compatible",
                    "DUOLA_API_BASE": "https://api.xiaomimimo.com/v1",
                },
                clear=False,
            ):
                config = load_runtime_config(config_path=Path(tmp) / "missing.json")

            provider = provider_config(config, config.agents.defaults.provider)
            self.assertEqual(provider.api_base, "https://api.xiaomimimo.com/v1")


if __name__ == "__main__":
    unittest.main()
