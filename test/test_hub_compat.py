import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch.hub import (
    _get_torch_home,
    download_url_to_file,
    load_state_dict_from_url,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestHubCompatibility(TestCase):
    def test_get_torch_home(self):
        with patch.dict(os.environ, {"TORCH_HOME": "/tmp/torch-home"}):
            self.assertEqual(_get_torch_home(), "/tmp/torch-home")

    def test_download_and_load_state_dict(self):
        state_dict = {"weight": torch.arange(4)}
        with tempfile.TemporaryDirectory() as source_dir:
            source = Path(source_dir) / "weights.pt"
            torch.save(state_dict, source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            with tempfile.TemporaryDirectory() as model_dir:
                downloaded = Path(model_dir) / f"weights-{digest[:8]}.pt"
                download_url_to_file(
                    source.as_uri(),
                    str(downloaded),
                    hash_prefix=digest[:8],
                    progress=False,
                )
                self.assertEqual(torch.load(downloaded, weights_only=True), state_dict)

            with tempfile.TemporaryDirectory() as model_dir:
                loaded = load_state_dict_from_url(
                    source.as_uri(),
                    model_dir=model_dir,
                    progress=False,
                    weights_only=True,
                )
                self.assertEqual(loaded, state_dict)


if __name__ == "__main__":
    run_tests()
