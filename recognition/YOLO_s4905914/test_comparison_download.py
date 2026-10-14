import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import download_comparison_weights as downloader


class DownloadTests(unittest.TestCase):
    def test_existing_wrong_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.pt'
            destination.write_bytes(b'wrong')
            with self.assertRaises(ValueError):
                downloader.download(destination)
            self.assertEqual(destination.read_bytes(), b'wrong')

    def test_verified_download_and_cache_no_network(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.pt'
            content = b'synthetic official-checkpoint fixture'
            digest = hashlib.sha256(content).hexdigest()
            with patch.object(downloader, 'COMPARISON_SHA256', digest), patch.object(
                    downloader.urllib.request, 'urlopen', return_value=io.BytesIO(content)) as request:
                downloader.download(destination)
                downloader.download(destination)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(len(list(Path(directory).glob('*.part'))), 0)

    def test_bad_download_does_not_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'model.pt'
            with patch.object(downloader.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaises(ValueError):
                    downloader.download(destination)
            self.assertFalse(destination.exists())
            self.assertEqual(len(list(Path(directory).glob('*.part'))), 0)


if __name__ == '__main__':
    unittest.main()
