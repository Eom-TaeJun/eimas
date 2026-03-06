# lib/auto_paper_execution.py — Shim (패키지 호환 레이어)
# 실제 구현은 lib/trading/auto_paper_execution.py 로 이동됨.
from lib.trading.auto_paper_execution import *  # noqa: F401, F403
from lib.trading.auto_paper_execution import AutoPaperExecutionConfig  # noqa: F401
