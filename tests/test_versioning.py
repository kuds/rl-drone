"""Tests for the git version info helper."""

import importlib.util
import subprocess
import sys
from pathlib import Path

# Load versioning.py directly so tests do not depend on optional MuJoCo /
# robot_descriptions imports pulled in by ``rl_drone.utils.__init__``.
_VERSIONING_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "rl_drone" / "utils" / "versioning.py"
)
_spec = importlib.util.spec_from_file_location("rl_drone_versioning_under_test", _VERSIONING_PATH)
versioning = importlib.util.module_from_spec(_spec)
sys.modules["rl_drone_versioning_under_test"] = versioning
_spec.loader.exec_module(versioning)

get_git_version_info = versioning.get_git_version_info


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path) -> str:
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    _git(["config", "commit.gpgsign", "false"], path)
    _git(["config", "tag.gpgsign", "false"], path)
    (path / "file.txt").write_text("hello\n")
    _git(["add", "file.txt"], path)
    _git(["commit", "-q", "-m", "init"], path)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


class TestGetGitVersionInfo:
    def test_clean_repo(self, tmp_path):
        sha = _init_repo(tmp_path)

        info = get_git_version_info(tmp_path)

        assert info["available"] is True
        assert info["commit"] == sha
        assert info["short_commit"] == sha[:8]
        assert info["dirty"] is False
        assert info["run_id"] == sha[:8]
        assert info["branch"] == "main"
        assert "dirty_files" not in info

    def test_dirty_repo_marks_run_id_and_lists_files(self, tmp_path):
        sha = _init_repo(tmp_path)
        (tmp_path / "file.txt").write_text("changed\n")
        (tmp_path / "new.txt").write_text("new\n")

        info = get_git_version_info(tmp_path)

        assert info["dirty"] is True
        assert info["run_id"] == f"{sha[:8]}-dirty"
        assert set(info["dirty_files"]) == {"file.txt", "new.txt"}

    def test_non_git_directory_reports_unavailable(self, tmp_path):
        info = get_git_version_info(tmp_path)

        assert info == {"available": False}
