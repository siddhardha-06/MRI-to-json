"""NeuroSphere v2: compact, verifiable JSON representations of T1-weighted brain MRI.

Phase 1 state: the package skeleton. Implemented: io (RAS loading), provenance, splits,
features.volumes_from_labels, qc (status only), jsonio (interim JSON), report.template
(measured-values-only report). preprocess, segment, normative and serialize arrive in
Phases 2-3 and raise NotImplementedError until then.
"""

__version__ = "2.0.0.dev1"
