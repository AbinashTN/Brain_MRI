import json

from src.utils import save_json


def test_save_json_creates_parent_and_writes(tmp_path):
    dest = tmp_path / "sub" / "file.json"
    save_json(dest, {"a": 1})
    assert json.loads(dest.read_text()) == {"a": 1}
