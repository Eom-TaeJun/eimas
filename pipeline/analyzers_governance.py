# pipeline/analyzers_governance.py — Shim (패키지 호환 레이어)
# 실제 구현은 pipeline/analyzers/governance.py 로 이동됨.
from pipeline.analyzers.governance import *  # noqa: F401, F403
from pipeline.analyzers.governance import _validation_input_fingerprint  # noqa: F401
