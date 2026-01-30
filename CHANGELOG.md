# Changelog

All notable changes to the EIMAS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] - 2026-01-30

### Added
- **lib/ Submodule Structure**: Organized lib/ into logical subdirectories
  - `lib/collectors/` - Data collection modules with BaseCollector interface
  - `lib/analyzers/` - Analysis engines with BaseAnalyzer interface
  - `lib/reports/` - Report generation modules
  - `lib/strategies/` - Portfolio strategy modules
  - `lib/db/` - Database interface modules
  - `lib/utils/` - Utility functions
- **Base Interfaces**: Created abstract base classes for extensibility
  - `BaseCollector` with Template Method pattern
  - `BaseAnalyzer` with Strategy pattern
- **Unified Archive Directory**: Consolidated scattered archive folders
  - `archive/legacy/` - Legacy main files
  - `archive/deprecated/` - Deprecated modules
  - `archive/agents/` - Archived agent implementations
  - `archive/core/` - Archived core modules
  - `archive/pipeline/` - Archived pipeline code
- **Documentation**:
  - `ARCHITECTURE.md` - Comprehensive architecture documentation
  - `CONTRIBUTING.md` - Developer contribution guidelines
  - This `CHANGELOG.md` file

### Changed  
- **main.py**: Enhanced with comprehensive bilingual docstrings
  - Added architecture overview in module docstring
  - Added pipeline phase diagram
  - Improved `run_integrated_pipeline()` documentation
- **core/schemas.py**: Enhanced header with schema organization explanation
- **pipeline/schemas.py**: Enhanced header with usage examples and class listing
- **lib/__init__.py**: Updated to version 2.2.0 with submodule exports

### Removed
- Moved `main_legacy.py` to `archive/legacy/`
- Moved `main_integrated.py` to `archive/legacy/`
- Cleaned up empty archive directories from lib/, agents/, core/, pipeline/

### Deprecated
- Direct imports from `lib/deprecated/` (moved to `archive/deprecated/`)

---

## [2.1.3] - 2026-01-29

### Added
- **Sentiment Analysis**: `analyze_sentiment()` function in pipeline
- **AI Report Integration**: `ai_report` field in EIMASResult
- **Bubble Risk Analysis**: `analyze_bubble_risk()` in pipeline analyzers
- **Multi-LLM Validation**: `run_ai_validation()` for cross-LLM consensus

### Changed
- Enhanced debate system with Phase 3 enhanced fields
- Updated pipeline/__init__.py with new analyzer exports
- Improved EIMASResult with reasoning_chain tracking

---

## [2.1.0] - 2026-01-25

### Added
- **Enhanced Analysis Suite**:
  - `analyze_hft_microstructure()` - HFT analysis
  - `analyze_volatility_garch()` - GARCH volatility modeling
  - `analyze_information_flow()` - Information asymmetry detection
  - `calculate_proof_of_index()` - Index verification
  - `analyze_dtw_similarity()` - Dynamic Time Warping similarity
  - `detect_outliers_with_dbscan()` - Outlier detection
  - `analyze_ark_trades()` - ARK Invest tracking

### Changed
- Updated EIMASResult schema with new analysis fields
- Enhanced to_markdown() method with new sections

---

## [2.0.0] - 2026-01-20

### Added
- **Modular Pipeline Architecture**: Complete refactoring to modular design
  - `pipeline/` package with collectors, analyzers, storage, report modules
  - `EIMASResult` unified result schema
- **Multi-Agent Debate System**:
  - InterpretationDebateAgent (economic schools)
  - MethodologyDebateAgent (statistical methods)
- **Dual Mode Analysis**: FULL vs REFERENCE historical data weighting

### Changed
- Migrated from monolithic main.py to modular pipeline
- Separated agent communication schemas to core/schemas.py
- Moved result schemas to pipeline/schemas.py

---

## [1.0.0] - 2026-01-15

### Added
- Initial release of EIMAS
- Basic pipeline with FRED data collection
- RegimeDetector with GMM classification
- CriticalPath risk scoring
- LASSO forecast agent
- HTML dashboard generation
- Basic CLI interface

---

## Future Plans (미구현 계획)

### [3.0.0] - Planned
- [ ] Web dashboard real-time updates
- [ ] Mobile companion app
- [ ] Comprehensive backtesting engine
- [ ] Push notification alert system
- [ ] Multi-language report generation
- [ ] Advanced portfolio rebalancing

---

*For more details on each release, see the corresponding git tags.*
