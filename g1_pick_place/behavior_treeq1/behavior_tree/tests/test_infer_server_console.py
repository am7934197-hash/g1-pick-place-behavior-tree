"""infer_server console tee writes to a file without swallowing stdout."""

import io
import sys
import tempfile
import unittest
from pathlib import Path

from infer_server import _TeeStream, attach_console_file_log


class TeeStreamTest(unittest.TestCase):
    def test_write_goes_to_both(self):
        primary = io.StringIO()
        log = io.StringIO()
        tee = _TeeStream(primary, log)
        tee.write("hello\n")
        tee.flush()
        self.assertEqual(primary.getvalue(), "hello\n")
        self.assertEqual(log.getvalue(), "hello\n")

    def test_attach_restores_and_appends(self):
        old_out, old_err = sys.stdout, sys.stderr
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "infer_server_console.log"
                attach_console_file_log(path)
                print("tee-line")
                sys.stderr.write("err-line\n")
                sys.stderr.flush()
                text = path.read_text(encoding="utf-8")
                self.assertIn("tee-line", text)
                self.assertIn("err-line", text)
        finally:
            sys.stdout, sys.stderr = old_out, old_err


if __name__ == "__main__":
    unittest.main()
