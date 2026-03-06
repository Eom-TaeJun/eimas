# lib/event_db.py — Shim (패키지 호환 레이어)
# 실제 구현은 lib/db/event_db.py 로 이동됨.
from lib.db.event_db import *  # noqa: F401, F403
from lib.db.event_db import EventDatabase, AutoSaveSession  # noqa: F401
# 하위 호환용 alias
EventDB = EventDatabase
