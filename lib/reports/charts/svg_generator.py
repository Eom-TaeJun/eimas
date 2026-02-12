"""
SVG Chart Generator
===================
SVG-based chart generation for PDF-compatible HTML reports.

Extracted from lib.final_report_agent for better modularity.
"""

from typing import List


def generate_svg_pie_chart(data: List[tuple], size: int = 160, hole_size: int = 60, center_text: str = "") -> str:
    """
    SVG 기반 파이 차트 생성 (PDF 변환 호환)

    Args:
        data: [(label, value, color), ...] 형식의 데이터
        size: 차트 크기 (px)
        hole_size: 도넛 홀 크기 (px), 0이면 일반 파이
        center_text: 중앙 텍스트

    Returns:
        SVG HTML 문자열
    """
    import math

    if not data:
        return '<div style="text-align: center; color: #868e96;">No data</div>'

    total = sum(v for _, v, _ in data)
    if total == 0:
        return '<div style="text-align: center; color: #868e96;">No data</div>'

    cx, cy = size / 2, size / 2
    r = (size - 10) / 2  # 약간의 여백

    paths = []
    start_angle = -90  # 12시 방향에서 시작

    for label, value, color in data:
        if value <= 0:
            continue

        pct = value / total
        end_angle = start_angle + (pct * 360)

        # 각도를 라디안으로 변환
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # 시작점과 끝점 계산
        x1 = cx + r * math.cos(start_rad)
        y1 = cy + r * math.sin(start_rad)
        x2 = cx + r * math.cos(end_rad)
        y2 = cy + r * math.sin(end_rad)

        # 큰 호 플래그 (180도 이상이면 1)
        large_arc = 1 if pct > 0.5 else 0

        # SVG path
        if hole_size > 0:
            # 도넛 차트
            inner_r = hole_size / 2
            ix1 = cx + inner_r * math.cos(start_rad)
            iy1 = cy + inner_r * math.sin(start_rad)
            ix2 = cx + inner_r * math.cos(end_rad)
            iy2 = cy + inner_r * math.sin(end_rad)

            path = f'M {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f} L {ix2:.2f} {iy2:.2f} A {inner_r:.2f} {inner_r:.2f} 0 {large_arc} 0 {ix1:.2f} {iy1:.2f} Z'
        else:
            # 일반 파이 차트
            path = f'M {cx:.2f} {cy:.2f} L {x1:.2f} {y1:.2f} A {r:.2f} {r:.2f} 0 {large_arc} 1 {x2:.2f} {y2:.2f} Z'

        paths.append(f'<path d="{path}" fill="{color}" stroke="#fff" stroke-width="1"/>')
        start_angle = end_angle

    # 중앙 텍스트
    center_html = ""
    if center_text and hole_size > 0:
        center_html = f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="700" fill="#212529">{center_text}</text>'

    svg = f'''<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
        {''.join(paths)}
        {center_html}
    </svg>'''

    return svg
