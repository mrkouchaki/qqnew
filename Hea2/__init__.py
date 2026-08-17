"""Read-only evaluation tools for the deployed PSAP KPI anomaly detector."""

from .synthetic_faults import (
    FaultSpec,
    InjectionResult,
    default_fault_suite,
    inject_faults,
)
from .production_scorer import ProductionScorer, ScoreResult
from .model_health_check import (
    HealthCheckConfig,
    evaluate_model_health,
    run_health_checks,
)

__all__ = [
    "FaultSpec",
    "InjectionResult",
    "default_fault_suite",
    "inject_faults",
    "ProductionScorer",
    "ScoreResult",
    "HealthCheckConfig",
    "evaluate_model_health",
    "run_health_checks",
]
