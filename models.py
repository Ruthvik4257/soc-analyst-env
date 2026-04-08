from typing import List, Optional, Literal
from openenv.core.env_server import Action, Observation, State

class SocAction(Action):
    action_type: Literal["search_logs", "get_threat_intel", "get_asset_info", "take_action"]
    query: Optional[str] = None
    time_range: Optional[str] = None
    indicator: Optional[str] = None
    hostname_or_user: Optional[str] = None
    decision: Optional[Literal["false_positive", "escalate_tier2", "block_if_malicious"]] = None
    reason: Optional[str] = None

class SocObservation(Observation):
    message: str
    remaining_steps: int
    alert_details: dict
    evidence_collected: List[str]
    score: Optional[float] = None

class SocState(State):
    difficulty: str = "easy"
    alert: dict = {}
    remaining_steps: int = 10
    evidence_collected: List[str] = []
    # Ground truth answers for grading
    expected_decision: str = ""
    # Whether episode ended successfully
    score: float = 0.01
