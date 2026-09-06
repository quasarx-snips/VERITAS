"""Serializable contracts for the deterministic downstream safety gate."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Mapping


class GateAction(str, Enum):
    """Actions available to downstream consumers."""

    ALLOW = "ALLOW"
    RESTRICT = "RESTRICT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class GatePolicy:
    """Configurable, verdict-to-action policy with no language-model inputs."""

    verdict_actions: Mapping[str, GateAction] = field(default_factory=lambda: {
        "STRONG": GateAction.ALLOW,
        "PARTIAL": GateAction.RESTRICT,
        "WEAK": GateAction.HUMAN_REVIEW,
        "NONE": GateAction.BLOCK,
    })

    def action_for(self, verdict: str) -> GateAction:
        try:
            action = self.verdict_actions[verdict]
        except KeyError as exc:
            raise ValueError(f"No gate action configured for verdict {verdict!r}") from exc
        return action if isinstance(action, GateAction) else GateAction(action)

    def to_dict(self) -> Dict[str, str]:
        return {name: self.action_for(name).value for name in self.verdict_actions}


@dataclass(frozen=True)
class GateResult:
    """A deterministic action and machine-readable decision reasons."""

    action: GateAction
    reasons: List[str]
    verdict: str
    policy: Mapping[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action.value,
            "reasons": list(self.reasons),
            "verdict": self.verdict,
            "policy": dict(self.policy),
        }
