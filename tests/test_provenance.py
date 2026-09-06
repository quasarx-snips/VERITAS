"""Tests for portable audit provenance."""

from veritas.audit import Provenance


def test_provenance_captures_configuration_dimensions_and_portable_names():
    provenance = Provenance.create(
        source_input=r"C:\Users\bijan\private\source.png",
        reference_input=r"C:\Users\bijan\private\reference.png",
        source_shape=(120, 240), reference_shape=(100, 200),
        configurations={"spatial_grid": {"grid_shape": (4, 4)}},
        run_id="test-run", timestamp="2026-01-01T00:00:00+00:00",
    )
    data = provenance.to_dict()
    assert data["run_id"] == "test-run"
    assert data["image_dimensions"]["source"] == {"height": 120, "width": 240}
    assert data["input_identifiers"]["source"] == "source.png"
    assert "C:\\Users" not in str(data)
