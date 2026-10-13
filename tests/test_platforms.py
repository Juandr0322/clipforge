import json
import shutil

import pytest

from clipforge.platforms import (
    PROFILES_DIR, ProfileError, load_all, load_profile, platform_names, shared_safe_zone,
)

MAIN = {"tiktok", "reels", "shorts"}
SIDE = {"x", "facebook"}


@pytest.fixture
def pdir(tmp_path):
    d = tmp_path / "profiles"
    shutil.copytree(PROFILES_DIR, d)
    return d


def _edit(d, name, fn):
    p = d / f"{name}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    fn(data)
    p.write_text(json.dumps(data), encoding="utf-8")


def test_loads_all_five_profiles():
    profiles = load_all()
    assert set(profiles) == MAIN | SIDE
    for name, p in profiles.items():
        assert p.name == name
        assert (p.width, p.height) == (1080, 1920)
        assert p.tier == ("main" if name in MAIN else "side")
        assert 0 < p.target_min_s <= p.target_max_s <= p.max_length_s
        assert p.checked and p.source


@pytest.mark.parametrize("mutate,field", [
    (lambda d: d["target_length_s"].update(max=d["max_length_s"] + 1), "target_length_s.max"),
    (lambda d: d["safe_zone"].update(x=900, width=500), "safe_zone"),
    (lambda d: d["safe_zone"].update(y=-5), "safe_zone"),
    (lambda d: d.pop("max_length_s"), "max_length_s"),
    (lambda d: d["safe_zone"].pop("height"), "safe_zone.height"),
    (lambda d: d.update(tier="primary"), "tier"),
])
def test_invalid_profile_names_file_and_field(pdir, mutate, field):
    _edit(pdir, "reels", mutate)
    with pytest.raises(ProfileError) as e:
        load_profile("reels", pdir)
    assert "reels.json" in str(e.value) and f"'{field}'" in str(e.value)


def test_malformed_json_names_file(pdir):
    (pdir / "x.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(ProfileError, match=r"x\.json"):
        load_profile("x", pdir)


def test_unknown_platform_lists_valid_names():
    with pytest.raises(ProfileError) as e:
        load_profile("myspace")
    msg = str(e.value)
    assert "myspace" in msg
    for n in MAIN | SIDE:
        assert n in msg
    assert set(platform_names()) == MAIN | SIDE


def test_shared_safe_zone_inside_frame_and_follows_profiles(pdir):
    z = shared_safe_zone()
    assert z.width > 0 and z.height > 0
    assert z.x >= 0 and z.y >= 0 and z.x + z.width <= 1080 and z.y + z.height <= 1920
    for p in load_all().values():
        if p.tier == "main":
            s = p.safe_zone
            assert z.x >= s.x and z.y >= s.y
            assert z.x + z.width <= s.x + s.width and z.y + z.height <= s.y + s.height

    base = shared_safe_zone(pdir)
    assert base == z
    _edit(pdir, "shorts", lambda d: d["safe_zone"].update(y=400, height=900))
    changed = shared_safe_zone(pdir)
    assert changed != base and changed.y == 400


def test_side_tier_does_not_affect_shared_zone(pdir):
    base = shared_safe_zone(pdir)
    _edit(pdir, "x", lambda d: d["safe_zone"].update(x=500, width=500))
    assert shared_safe_zone(pdir) == base
