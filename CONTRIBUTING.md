# Contributing to EIMAS

Thank you for your interest in contributing to EIMAS! This document provides guidelines and best practices for contributing to the project.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style Guide](#code-style-guide)
4. [Adding New Features](#adding-new-features)
5. [Pull Request Process](#pull-request-process)
6. [Testing Guidelines](#testing-guidelines)

---

## 🚀 Getting Started

### Mandatory First Read

- Before any implementation/refactor, read `command.md` first.
- Treat `command.md` as the source of truth for entrypoint and command policy.
- New independent features must be attached to `main.py` flags (`python main.py --abc`) instead of adding new runner scripts.
- Follow the document priority tiers (`P0~P3`) defined in `command.md`.

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Git

### Clone and Setup

```bash
# Clone repository
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

---

## 💻 Development Setup

### Running the Backend

```bash
# Main pipeline (quick mode)
python main.py --short

# API server
python api/main.py
```

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:3002
```

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_lib.py -v

# With coverage
python -m pytest --cov=lib tests/
```

---

## 📝 Code Style Guide

### Python Style

We follow PEP 8 with some modifications:

```python
# Good: Descriptive function names with docstrings
def analyze_bubble_risk(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze bubble risk indicators for given market data.
    버블 리스크 지표 분석.
    
    Args:
        market_data: Dictionary with ticker keys and DataFrame values
        
    Returns:
        Dict containing bubble risk metrics
    """
    # Implementation...
    pass

# Good: Type hints
def calculate_vpin(
    trades: pd.DataFrame,
    bucket_size: int = 50
) -> Tuple[float, List[float]]:
    ...

# Good: Constants at module level
DEFAULT_LOOKBACK_DAYS = 365
MAX_RISK_SCORE = 100
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | `snake_case` | `analyze_regime()` |
| Classes | `PascalCase` | `RegimeDetector` |
| Constants | `UPPER_SNAKE` | `MAX_DRAWDOWN` |
| Private | `_leading_underscore` | `_calculate_internal()` |

### Bilingual Comments

For key functions, provide bilingual docstrings:

```python
def detect_regime(prices: pd.DataFrame) -> RegimeResult:
    """
    Detect current market regime using GMM.
    GMM을 사용한 시장 레짐 탐지.
    
    Args:
        prices: OHLCV price data / 가격 데이터
        
    Returns:
        RegimeResult with classification / 레짐 분류 결과
    """
```

---

## ➕ Adding New Features

### Adding a New Analyzer

1. Create the analyzer in `lib/`:

```python
# lib/my_analyzer.py
from lib.analyzers.base import BaseAnalyzer

class MyAnalyzer(BaseAnalyzer):
    """My custom analyzer."""
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis."""
        # Implementation
        return {"result": ...}
    
    def get_summary(self) -> str:
        return "My analysis summary"
    
    def get_analyzer_name(self) -> str:
        return "MyAnalyzer"
```

2. Export in `lib/analyzers/__init__.py`:

```python
from lib.my_analyzer import MyAnalyzer
__all__ = [..., 'MyAnalyzer']
```

3. Integrate into pipeline (`pipeline/analyzers.py`):

```python
def analyze_my_feature(market_data: Dict) -> Dict:
    analyzer = MyAnalyzer()
    return analyzer.analyze(market_data)
```

4. Add to `pipeline/__init__.py`:

```python
from .analyzers import analyze_my_feature
__all__ = [..., 'analyze_my_feature']
```

### Adding to EIMASResult

When adding new analysis outputs, update `pipeline/schemas.py`:

```python
@dataclass
class EIMASResult:
    # ... existing fields ...
    
    # NEW: My feature analysis
    my_feature_result: Dict = field(default_factory=dict)
```

---

## 🔄 Pull Request Process

### PR Checklist

- [ ] Code follows style guide
- [ ] Added/updated docstrings (bilingual for key functions)
- [ ] Added unit tests for new features
- [ ] All tests pass locally
- [ ] Updated relevant documentation
- [ ] No sensitive data (API keys, etc.)

### PR Title Format

```
[CATEGORY] Brief description

Examples:
[FEAT] Add sector momentum analyzer
[FIX] Fix regime detection edge case
[DOCS] Update architecture documentation
[REFACTOR] Modularize lib/ structure
```

### Branch Naming

```
feature/analyzer-sector-momentum
fix/regime-detection-nan
docs/update-architecture
refactor/lib-structure
```

---

## 🧪 Testing Guidelines

### Test File Structure

```
tests/
├── test_lib.py            # Library module tests
├── test_api_connection.py # API tests
├── test_integration.py    # Full pipeline tests
└── test_multi_agent.py    # Agent system tests
```

### Writing Tests

```python
import unittest
from lib.regime_detector import RegimeDetector

class TestRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RegimeDetector()
    
    def test_detect_regime_returns_valid_result(self):
        """Test that detect_regime returns a valid RegimeResult."""
        result = self.detector.detect_regime()
        
        self.assertIn(result.regime, ['BULL', 'BEAR', 'SIDEWAYS'])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_handles_empty_data_gracefully(self):
        """Test graceful handling of empty input."""
        result = self.detector.detect_regime(prices=pd.DataFrame())
        
        self.assertEqual(result.regime, 'UNKNOWN')
```

---

## 📞 Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: eimas-dev@example.com

---

*Thank you for contributing to EIMAS!*
