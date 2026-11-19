"""Pure domain types.

This package imports nothing from its siblings and touches no network, no disk
and no API. Everything here can be constructed in a test in one line.
"""

from lazyapply_core.domain.evidence import (
    EndDate,
    Evidence,
    EvidenceId,
    ScoredEvidence,
    YearMonth,
)
from lazyapply_core.domain.fit import (
    FitReport,
    MatchStatus,
    RequirementMatch,
    Verdict,
)
from lazyapply_core.domain.pack import EvidencePack, RequirementEvidence
from lazyapply_core.domain.profile import Education, Link, Position, Profile
from lazyapply_core.domain.role import (
    Requirement,
    RequirementCategory,
    RequirementKind,
    Role,
)

__all__ = [
    "Education",
    "EndDate",
    "Evidence",
    "EvidenceId",
    "EvidencePack",
    "FitReport",
    "Link",
    "MatchStatus",
    "Position",
    "Profile",
    "Requirement",
    "RequirementCategory",
    "RequirementEvidence",
    "RequirementKind",
    "RequirementMatch",
    "Role",
    "ScoredEvidence",
    "Verdict",
    "YearMonth",
]
