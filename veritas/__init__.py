"""VERITAS — correspondence verification and gating system.

Conceptual pipeline (fixed stage order):

    IMAGE PAIR
    -> preprocessing
    -> independent feature evidence
    -> descriptor matching
    -> geometric verification  (AFFINE-ONLY certification)
    -> spatial evidence
    -> counter-evidence
    -> evidence fusion
    -> verdict
    -> downstream gate
    -> audit report

The package is under active migration from proven primitives hosted in the
reference repository. Subpackages are imported by their owning modules; this
``__init__`` stays import-light until the first migration commits land.
"""

__version__ = "0.2.0"