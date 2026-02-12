"""
EIMAS Report Charts
===================
Chart generation utilities for HTML reports.

Available functions:
    - generate_svg_pie_chart: SVG-based pie/donut chart generator
"""

from .svg_generator import generate_svg_pie_chart

__all__ = ['generate_svg_pie_chart']
