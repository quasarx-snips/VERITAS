"""Bootstrap checks for the VERITAS repository skeleton.

These tests validate structure and policy only; algorithm behaviour is tested
by dedicated tests as each migration commit lands.

The forbidden-token scan protects the core protocol rule: the final VERITAS
implementation must never import from the source repository
(``sihcoremodule26166`` / ``lunax``) and must never mutate ``sys.path`` to
reach it.
"""

import pathlib
import re

import veritas
import veritas.config

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VERITAS_PACKAGE_DIR = REPO_ROOT / "veritas"

EXPECTED_SUBPACKAGES = (
    "preprocessing",
    "features",
    "matching",
    "geometry",
    "spatial",
    "counter_evidence",
    "verification",
    "verdict",
    "gate",
    "audit",
    "llm",
)

EXPECTED_TOP_LEVEL_DIRS = (
    "configs",
    "data",
    "docs",
    "scripts",
    "tests",
    "outputs",
    "demo",
)


def test_package_version_is_readable() -> None:
    assert isinstance(veritas.__version__, str) and veritas.__version__


def test_all_veritas_subpackages_are_importable() -> None:
    for name in EXPECTED_SUBPACKAGES:
        __import__(f"veritas.{name}")
        assert (VERITAS_PACKAGE_DIR / name / "__init__.py").is_file()


def test_core_modules_exist() -> None:
    for name in ("pipeline", "schemas", "config"):
        assert (VERITAS_PACKAGE_DIR / f"{name}.py").is_file()
        __import__(f"veritas.{name}")


def test_top_level_directories_exist() -> None:
    for name in EXPECTED_TOP_LEVEL_DIRS:
        assert (REPO_ROOT / name).is_dir(), f"missing {name}/"


def test_affine_only_certification_policy() -> None:
    assert veritas.config.CERTIFICATION_MODELS == ("affine",)
    assert "homography" not in veritas.config.CERTIFICATION_MODELS
    assert "auto" not in veritas.config.CERTIFICATION_MODELS


def test_doc_path_contains_migration_map() -> None:
    map_path = REPO_ROOT / "docs" / "source_migration_map.md"
    assert map_path.is_file()


def test_no_forbidden_call_into_source_repository() -> None:
    """The veritas package must not reach into the reference repository."""
    forbidden = re.compile(
        r"\b(sihcoremodule26166|lunax)\b|sys\.path\.(append|insert)"
    )
    offenders = []
    for path in VERITAS_PACKAGE_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "forbidden source-repository references found:\n" + "\n".join(offenders)