import os
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
os.environ["MONGODB_URI"] = ""
os.environ["MONGO_REQUIRED"] = "0"
os.environ["STORAGE_BACKEND"] = "local"

import app  # noqa: E402


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_users_path = app.LOCAL_USERS_PATH
        self.original_results_path = app.LOCAL_RESULTS_PATH
        self.original_mongo = app._MONGO_DB
        app._MONGO_DB = None
        root = pathlib.Path(self.temp_dir.name)
        app.LOCAL_USERS_PATH = root / "users.json"
        app.LOCAL_RESULTS_PATH = root / "results.json"

    def tearDown(self):
        app.LOCAL_USERS_PATH = self.original_users_path
        app.LOCAL_RESULTS_PATH = self.original_results_path
        app._MONGO_DB = self.original_mongo
        self.temp_dir.cleanup()

    def test_same_account_can_login_repeatedly(self):
        app.create_user("Listener", "listener@example.com", "correct-password")
        first = app.login(app.LoginPayload(email="listener@example.com", password="correct-password"))
        second = app.login(app.LoginPayload(email="LISTENER@example.com", password="correct-password"))
        self.assertEqual(first["user"]["id"], second["user"]["id"])
        self.assertNotEqual(first["token"], "")
        self.assertNotEqual(second["token"], "")

    def test_local_history_is_scoped_to_current_user(self):
        first = {"id": "first-user"}
        second = {"id": "second-user"}
        result = {"status": "success", "cache_id": "run", "files": ["/output/a.wav"], "stems": []}
        app.persist_result(first, result, "first.wav", ["Vocal"])
        app.persist_result(second, result, "second.wav", ["Drums"])
        history = app.list_persisted_results(first)
        self.assertEqual([item["inputName"] for item in history], ["first.wav"])

    def test_private_supabase_signed_path_contains_storage_prefix(self):
        with patch.object(app, "SUPABASE_URL", "https://project.supabase.co"), patch.object(
            app, "SUPABASE_PUBLIC_BUCKET", False
        ), patch.object(app, "supabase_request", return_value={"signedURL": "/object/sign/bucket/file.wav?token=x"}):
            url = app.create_supabase_url("users/u/run/file.wav")
        self.assertEqual(url, "https://project.supabase.co/storage/v1/object/sign/bucket/file.wav?token=x")

    def test_bucket_bootstrap_uses_configured_visibility(self):
        with patch.object(app, "SUPABASE_BUCKET", "stems"), patch.object(
            app, "SUPABASE_PUBLIC_BUCKET", False
        ), patch.object(app, "supabase_request", return_value={}) as request:
            app.ensure_supabase_bucket()
        body = request.call_args.kwargs["body"]
        self.assertEqual(json.loads(body), {"id": "stems", "name": "stems", "public": False})

    def test_stem_catalog_uses_final_unet_outputs(self):
        with patch.object(app, "find_model_path", return_value=pathlib.Path("model_render.pth")):
            payload = app.available_stems()
        self.assertFalse(payload["model_loaded"])
        self.assertEqual(payload["stems"][0], {"name": "Vocal", "label": "Vocals"})

    def test_inference_chunk_is_bounded_for_render(self):
        self.assertEqual(app.inference_chunk_samples(64000, 16000), 16000)
        self.assertEqual(app.inference_chunk_samples(8000, 16000), 8000)

    def test_frequency_tiles_cover_the_full_spectrogram(self):
        starts = app.frequency_tile_starts(1025, 512, 128)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1] + 512, 1025)
        self.assertTrue(all(right - left <= 384 for left, right in zip(starts, starts[1:])))


if __name__ == "__main__":
    unittest.main()
