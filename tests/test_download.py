import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_data import extract_csvs, load_manifest, main, publish


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DownloadTests(unittest.TestCase):
    def archive(self, files):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, content in files:
                archive.writestr(name, content)
        return stream.getvalue()

    def test_flattens_paths_and_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data"
            count = extract_csvs(self.archive([("../../coin.csv", "Date,Close\n")]), destination)
            self.assertEqual(count, 1)
            self.assertTrue((destination / "coin.csv").is_file())
            self.assertFalse((Path(directory) / "coin.csv").exists())

    def test_collisions_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with self.assertRaisesRegex(ValueError, "colliding"):
                extract_csvs(self.archive([("a/coin.csv", "a"), ("b/COIN.csv", "b")]), destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            (destination / "coin.csv").write_text("original")
            with self.assertRaises(FileExistsError):
                extract_csvs(self.archive([("coin.csv", "new")]), destination)
            self.assertEqual((destination / "coin.csv").read_text(), "original")


class DownloadMainTests(unittest.TestCase):
    """Exercise the documented ``--zip`` path with a temporary manifest."""

    FILES = [("sub/alpha.csv", "Date,Close\n2020-01-01,1\n"), ("beta.csv", "Date,Close\n2020-01-01,2\n")]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.payload = self.archive(self.FILES)
        self.zip = self.root / "archive.zip"
        self.zip.write_bytes(self.payload)
        self.output = self.root / "data" / "raw"

    def tearDown(self):
        self.directory.cleanup()

    def archive(self, files):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name, content in files:
                archive.writestr(name, content)
        return stream.getvalue()

    def manifest(self, archive_sha256=None, files=None):
        record = {
            "archive_sha256": archive_sha256 or sha256(self.payload),
            "files": files or {Path(name).name: sha256(content.encode()) for name, content in self.FILES},
        }
        path = self.root / "manifest.json"
        path.write_text(json.dumps(record))
        return path

    def run_main(self, manifest):
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            try:
                code = main(["--zip", str(self.zip), "--output", str(self.output), "--manifest", str(manifest)])
            except SystemExit as exit_:
                code = exit_.code
        return code, stderr.getvalue()

    def leftovers(self):
        parent = self.output.parent
        if not parent.exists():
            return []
        return sorted(p.name for p in parent.iterdir() if p.name.startswith(".crypto-download-"))

    def test_verified_zip_is_published_atomically(self):
        code, _ = self.run_main(self.manifest())
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["alpha.csv", "beta.csv"])
        self.assertEqual((self.output / "beta.csv").read_text(), "Date,Close\n2020-01-01,2\n")
        self.assertEqual(self.leftovers(), [])
        mask = os.umask(0)
        os.umask(mask)
        # The published directory must not keep mkdtemp's owner-only mode.
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o777 & ~mask)

    def test_changed_archive_is_refused_without_writing(self):
        code, message = self.run_main(self.manifest(archive_sha256="0" * 64))
        self.assertEqual(code, 1)
        self.assertIn("Verification failed", message)
        self.assertIn("differs from the recorded snapshot", message)
        self.assertNotIn("--zip PATH", message)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.leftovers(), [])

    def test_changed_file_hash_is_refused_without_writing(self):
        files = {"alpha.csv": "1" * 64, "beta.csv": sha256(self.FILES[1][1].encode())}
        code, message = self.run_main(self.manifest(files=files))
        self.assertEqual(code, 1)
        self.assertIn("CSV hashes do not match", message)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.leftovers(), [])

    def test_invalid_manifest_fails_before_reading_the_archive(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"files": {}}))
        # A missing ZIP would fail first if the archive were read before the manifest.
        self.zip.unlink()
        code, message = self.run_main(manifest)
        self.assertEqual(code, 1)
        self.assertIn("Reading the manifest failed", message)
        self.assertIn("'archive_sha256'", message)
        with self.assertRaisesRegex(ValueError, "archive_sha256"):
            load_manifest(manifest)
        manifest.write_text(json.dumps(["not", "an", "object"]))
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_manifest(manifest)

    def test_existing_csvs_in_output_are_refused(self):
        self.output.mkdir(parents=True)
        (self.output / "old.CSV").write_text("x")
        with self.assertRaises(SystemExit) as raised:
            with contextlib.redirect_stderr(io.StringIO()):
                main(["--zip", str(self.zip), "--output", str(self.output), "--manifest", str(self.manifest())])
        self.assertEqual(raised.exception.code, 2)

    def test_publish_into_existing_directory_checks_names_case_insensitively(self):
        staging = self.root / "staging"
        staging.mkdir()
        (staging / "alpha.csv").write_text("new")
        self.output.mkdir(parents=True)
        (self.output / "ALPHA.CSV").write_text("old")
        with self.assertRaisesRegex(FileExistsError, "alpha.csv"):
            publish(staging, self.output)
        self.assertEqual((self.output / "ALPHA.CSV").read_text(), "old")
        (self.output / "ALPHA.CSV").unlink()
        publish(staging, self.output)
        self.assertEqual((self.output / "alpha.csv").read_text(), "new")


if __name__ == "__main__":
    unittest.main()
