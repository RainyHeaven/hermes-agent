"""``session.resume`` cwd fallback (FB35-H): a stored cwd always wins; a session with none takes the opener's
folder (``params.cwd``, absolute + existing) before the profile default.
"""

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session


@pytest.fixture
def work(tmp_path):
    # Resolved: on macOS tmp_path sits under /var -> /private/var.
    root = tmp_path.resolve() / "work"
    (root / "proj").mkdir(parents=True)
    (root / "stored").mkdir()
    return root


def test_empty_stored_cwd_takes_params_cwd(work):
    assert srv._resume_fallback_cwd("", str(work / "proj"), None) == str(work / "proj")


def test_stored_cwd_beats_params_cwd(work):
    assert srv._resume_fallback_cwd(str(work / "stored"), str(work / "proj"), None) == str(work / "stored")


def test_bad_params_cwd_falls_back_to_profile_default(work):
    default = srv._profile_default_cwd(None)
    for bad in (str(work / "gone"), "work/proj", ""):
        assert srv._resume_fallback_cwd("", bad, None) == default
