"""VERITAS pipeline orchestration — architecture skeleton.

INCOMPLETE MODULE — the orchestration is not implemented yet.

The end-to-end correspondence verification chain is deliberately not wired here
until the P0 primitives (preprocessing, SIFT, matching, affine geometry,
verification metrics) have been migrated and tested individually. This module
must stay import-light: do not import veritas subpackages from here before
their first migration commits land.

Chain of stages (fixed order):
    image pair
      -> preprocessing
      -> independent feature evidence
      -> descriptor matching
      -> geometric verification (AFFINE-ONLY certification)
      -> spatial evidence
      -> counter-evidence
      -> evidence fusion
      -> verdict
      -> downstream gate
      -> audit report

Stage boundaries follow the migration map in ``docs/source_migration_map.md``.
"""