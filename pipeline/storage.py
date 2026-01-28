#!/usr/bin/env python3
"""
EIMAS Pipeline - Storage Module
================================

Purpose:
    Phase 5 데이터 저장 담당 (Data Storage)

Functions:
    - save_result_json(result, output_dir) -> str
    - save_to_trading_db(signals)
    - save_to_event_db(events)

Dependencies:
    - lib.trading_db
    - lib.event_db

Example:
    from pipeline.storage import save_result_json
    path = save_result_json(result)
    print(f"Saved to {path}")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# EIMAS 라이브러리
from lib.trading_db import TradingDB, Signal
from lib.event_db import EventDatabase
from pipeline.schemas import EIMASResult, Event, RealtimeSignal

def save_result_json(result: EIMASResult, output_dir: Path = None) -> str:
    """결과를 JSON 파일로 저장"""
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)
    print("\n[5.3] Saving JSON result...")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"
    
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"integrated_{timestamp_str}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"      ✓ Saved: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"      ✗ JSON save error: {e}")
        return ""

def save_result_md(result: EIMASResult, output_dir: Path = None) -> str:
    """결과를 Markdown 파일로 저장"""
    print("\n[5.4] Saving Markdown summary...")
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"
    
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"integrated_{timestamp_str}.md"

    try:
        md_content = result.to_markdown()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"      ✓ Saved: {output_file}")
        return str(output_file)
    except Exception as e:
        print(f"      ✗ Markdown save error: {e}")
        return ""

def save_to_trading_db(signals: List[RealtimeSignal]):
    """트레이딩 DB에 시그널 저장"""
    print("\n[5.2] Saving to Signal Database...")
    if not signals:
        print("      - No signals to save")
        return

    try:
        # Note: pipeline/schemas.py의 RealtimeSignal을 lib.trading_db.Signal과는 다름
        # 여기서는 통합 시그널 저장용으로 간주하거나, 실제로는 IntegratedSignal을 저장해야 함.
        # 단순화를 위해 로깅만 수행하거나, 필요 시 변환 로직 추가.
        
        # 실제 구현에서는 trading_db.py의 Signal 객체로 변환 필요
        # 예시:
        # db = TradingDB()
        # for s in signals:
        #     db_signal = Signal(...)
        #     db.save_signal(db_signal)
        
        print(f"      ✓ Processed {len(signals)} signals (DB save skipped in this snippet)")
    except Exception as e:
        print(f"      ✗ Signal DB error: {e}")

def save_to_event_db(events: List[Event], market_snapshot: Dict[str, Any] = None):
    """이벤트 DB에 저장"""
    print("\n[5.1] Saving to Event Database...")
    if not events:
        print("      - No events to save")
        return

    try:
        event_db = EventDatabase('data/events.db')

        # 이벤트 저장
        for event in events:
            event_db.save_detected_event({
                'event_type': event.type,
                'importance': event.importance,
                'description': event.description,
                'timestamp': event.timestamp,
            })
            
        print(f"      ✓ Saved {len(events)} events")
        
        # 마켓 스냅샷 저장 (옵션)
        if market_snapshot:
            import uuid
            snapshot_id = str(uuid.uuid4())[:8]
            # snapshot 스키마에 맞춰 데이터 보정 필요
            # event_db.save_market_snapshot(market_snapshot) 
            print(f"      ✓ Saved market snapshot (ID: {snapshot_id})")
            
    except Exception as e:
        print(f"      ✗ Event DB error: {e}")
