"""File labels must name one file each.

A framework has several files with the same basename -- optim/base.py and
config/base.py, one registry.py per package. Labelling by basename gave the
first one the bare name and sent every request for the others to it: VIEW
optim/base.py returned config/base.py under a header reading "base.py", and a
fix targeting optim/base.py was applied against config/base.py.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pulse.pulse_cli import PulseCLI  # noqa: E402


def make_cli(root, files, entry="train.py"):
    """A CLI that knows `files` (relative paths) without running any setup."""
    def write(rel, text):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    cli = PulseCLI.__new__(PulseCLI)
    cli.script_path = write(entry, "# train.py\nVALUE = 'train.py'\n")
    cli.extra_files = {write(rel, f"# {rel}\nVALUE = {rel!r}\n"): f"# {rel}\nVALUE = {rel!r}\n"
                       for rel in files}
    cli.code_text = "# train.py\nVALUE = 'train.py'\n"
    cli._build_file_labels()
    return cli


class FileLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.cli = make_cli(self.root, [
            "config/base.py", "optim/base.py", "layers/base.py",
            "data/registry.py", "optim/registry.py", "models/model.py",
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_file_has_its_own_label(self):
        labels = sorted(self.cli._path_for_label)
        self.assertEqual(labels, ["config/base.py", "data/registry.py", "layers/base.py",
                                  "models/model.py", "optim/base.py", "optim/registry.py",
                                  "train.py"])

    def test_a_shared_basename_resolves_to_the_file_that_was_asked_for(self):
        for rel in ("optim/base.py", "config/base.py", "layers/base.py", "optim/registry.py"):
            self.assertEqual(self.cli._resolve_fix_path(rel), os.path.join(self.root, rel))

    def test_view_returns_the_file_that_was_asked_for(self):
        out = self.cli._run_view("optim/base.py:1-2")
        self.assertIn("optim/base.py", out.splitlines()[0])
        self.assertIn("optim/base.py", out)
        self.assertNotIn("config/base.py", out)

    def test_a_longer_path_than_the_label_still_resolves(self):
        self.assertEqual(self.cli._resolve_fix_path("mlkit/optim/base.py"),
                         os.path.join(self.root, "optim", "base.py"))

    def test_an_ambiguous_name_resolves_to_nothing_rather_than_to_the_wrong_file(self):
        self.assertIsNone(self.cli._resolve_fix_path("base.py"))
        self.assertIsNone(self.cli._resolve_fix_path("registry.py"))

    def test_an_unambiguous_basename_still_resolves(self):
        self.assertEqual(self.cli._resolve_fix_path("model.py"),
                         os.path.join(self.root, "models", "model.py"))

    def test_no_file_label_means_the_entry_script(self):
        self.assertEqual(self.cli._resolve_fix_path(""), self.cli.script_path)
        self.assertEqual(self.cli._resolve_fix_path(None), self.cli.script_path)

    def test_a_flat_project_keeps_plain_names(self):
        with tempfile.TemporaryDirectory() as flat:
            cli = make_cli(flat, ["model.py", "utils.py"])
            self.assertEqual(sorted(cli._path_for_label), ["model.py", "train.py", "utils.py"])

    def test_a_single_file_project_labels_the_script_by_name(self):
        with tempfile.TemporaryDirectory() as solo:
            cli = make_cli(solo, [])
            self.assertEqual(list(cli._path_for_label), ["train.py"])
            self.assertEqual(cli._resolve_fix_path("train.py"), cli.script_path)

    def test_files_above_the_script_are_labelled_from_the_shared_root(self):
        with tempfile.TemporaryDirectory() as root:
            cli = make_cli(root, ["lib/helpers.py"], entry="experiments/train.py")
            self.assertEqual(sorted(cli._path_for_label), ["experiments/train.py", "lib/helpers.py"])
            self.assertEqual(cli._resolve_fix_path("lib/helpers.py"),
                             os.path.join(root, "lib", "helpers.py"))

    def test_grep_reports_the_label_that_view_accepts(self):
        out = self.cli._run_grep("VALUE")
        for line in out.splitlines():
            if line.strip().startswith("optim/base.py"):
                label = line.strip().split(",")[0]
                self.assertEqual(self.cli._resolve_fix_path(label),
                                 os.path.join(self.root, "optim", "base.py"))
                break
        else:
            self.fail("optim/base.py was not reported by GREP")


if __name__ == "__main__":
    unittest.main()
