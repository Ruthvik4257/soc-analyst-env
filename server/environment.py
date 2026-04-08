import uuid
import sys
import os
from typing import Optional
from openenv.core.env_server import Environment

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import SocAction, SocObservation, SocState

class SocAnalystEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 10

    def __init__(self):
        self._state = SocState()

    def reset(self, seed=None, episode_id=None, **kwargs) -> SocObservation:
        difficulty = kwargs.get("difficulty", "easy")
        
        # Reset state completely
        self._state = SocState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty,
            remaining_steps=self.MAX_STEPS,
            evidence_collected=[],
            score=0.01
        )

        if difficulty == "easy":
            self._state.alert = {"id": "A-001", "type": "brute_force", "ip": "203.0.113.5", "target": "10.0.0.5"}
            self._state.expected_decision = "block_if_malicious"
        elif difficulty == "medium":
            self._state.alert = {"id": "A-002", "type": "malware_detected", "hash": "Emotet", "target": "finance-laptop-01"}
            self._state.expected_decision = "escalate_tier2"
        elif difficulty == "hard":
            self._state.alert = {"id": "A-003", "type": "impossible_travel", "user": "jsmith", "ip": "198.51.100.2"}
            self._state.expected_decision = "false_positive"
        else:
            self._state.alert = {"id": "A-00x", "type": "unknown"}
            self._state.expected_decision = "false_positive"

        return SocObservation(
            done=False,
            reward=0.0,
            message=f"New alert received on difficulty: {difficulty}. Investigate and take_action.",
            remaining_steps=self._state.remaining_steps,
            alert_details=self._state.alert,
            evidence_collected=self._state.evidence_collected,
            score=self._state.score
        )

    def step(self, action: SocAction, timeout_s=None, **kwargs) -> SocObservation:
        self._state.step_count += 1
        self._state.remaining_steps -= 1
        
        at = action.action_type
        reward = 0.0
        message = ""
        done = False

        if at == "search_logs":
            if action.query == "203.0.113.5" and self._state.difficulty == "easy":
                message = "Logs for 203.0.113.5: 15 failed logins, 0 successful."
                reward = 0.1
                if "logs_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("logs_checked")
            elif action.query == "jsmith" and self._state.difficulty == "hard":
                message = "Logs for jsmith: successful login via VPN."
                reward = 0.1
                if "logs_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("logs_checked")
            else:
                message = f"No useful logs found for '{action.query}'."
                
        elif at == "get_threat_intel":
            if action.indicator == "203.0.113.5" and self._state.difficulty == "easy":
                message = "Threat intel for 203.0.113.5: Known Malicious IP (Botnet)."
                reward = 0.2
                if "ti_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("ti_checked")
            elif action.indicator == "Emotet" and self._state.difficulty == "medium":
                message = "Threat intel for Emotet: High severity trojan malware."
                reward = 0.2
                if "ti_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("ti_checked")
            elif action.indicator == "198.51.100.2" and self._state.difficulty == "hard":
                message = "Threat intel for 198.51.100.2: IP belongs to a Zscaler Corporate Gateway."
                reward = 0.3
                if "ti_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("ti_checked")
            else:
                message = f"Threat intel: No data on '{action.indicator}'."
                
        elif at == "get_asset_info":
            if action.hostname_or_user == "finance-laptop-01" and self._state.difficulty == "medium":
                message = "Asset finance-laptop-01: Owner CFO, Criticality High."
                reward = 0.2
                if "asset_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("asset_checked")
            elif action.hostname_or_user == "jsmith" and self._state.difficulty == "hard":
                message = "User jsmith: Department Sales, traveling status: unknown."
                reward = 0.1
                if "asset_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("asset_checked")
            else:
                message = f"No asset info found for '{action.hostname_or_user}'."
                
        elif at == "take_action":
            done = True
            if action.decision == self._state.expected_decision:
                if self._state.difficulty == "hard" and "ti_checked" not in self._state.evidence_collected:
                    message = f"You guessed {action.decision} but didn't check threat intel to verify! Score: 0.1."
                    reward = 0.1
                    self._state.score = 0.1
                else:
                    message = f"Correctly resolved alert as {action.decision}."
                    reward = 0.99
                    self._state.score = 0.99
            else:
                message = f"Incorrect decision: {action.decision}. Expected: {self._state.expected_decision}."
                reward = -0.5
                # Keep task scores strictly within (0, 1) for OpenEnv validators.
                self._state.score = 0.01
        
        else:
            message = f"Unknown action type: {at}"

        if not done and self._state.remaining_steps <= 0:
            done = True
            message = "Max steps reached without a decision. You failed to triage the alert."
            reward = 0.0
            self._state.score = 0.01
            
        return SocObservation(
            message=message,
            remaining_steps=self._state.remaining_steps,
            alert_details=self._state.alert,
            evidence_collected=self._state.evidence_collected,
            score=self._state.score,
            done=done,
            reward=reward
        )

    @property
    def state(self) -> SocState:
        return self._state
