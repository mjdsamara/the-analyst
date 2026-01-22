"""
Middleware package for The Analyst platform.

Provides runtime enforcement of BORIS compliance features:
- Autonomy level enforcement
- Cost tracking and budget alerts
- Comprehensive audit logging
"""

from src.middleware.audit import AuditEvent, AuditLogger, AuditMiddleware
from src.middleware.autonomy import (
    AutonomyConfig,
    AutonomyMiddleware,
    RestrictedToolCategory,
)
from src.middleware.cost_tracking import (
    CostEntry,
    CostTracker,
    CostTrackingMiddleware,
    SessionCostSummary,
)

__all__ = [
    # Autonomy
    "AutonomyMiddleware",
    "AutonomyConfig",
    "RestrictedToolCategory",
    # Cost Tracking
    "CostTracker",
    "CostTrackingMiddleware",
    "CostEntry",
    "SessionCostSummary",
    # Audit
    "AuditLogger",
    "AuditMiddleware",
    "AuditEvent",
]
