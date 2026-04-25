import os
import sys
import uuid
from typing import Dict, List

from openenv.core.env_server import Environment

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import AgentMessage, AgentObservation, SocAction, SocObservation, SocState
from server.datasets import search_uploaded_logs
from server.integrations import get_splunk_client

class SocAnalystEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_STEPS = 10
    TURN_ORDER = ["supervisor", "log_hunter", "threat_intel", "supervisor"]

    def __init__(self):
        self._state = SocState()

    def _resolve_expected_decision(self) -> str:
        """Recover expected decision if state is missing it."""
        if self._state.expected_decision:
            return self._state.expected_decision

        alert_type = (self._state.alert or {}).get("type")
        if alert_type == "brute_force":
            return "block_if_malicious"
        if alert_type == "malware_detected":
            return "escalate_tier2"
        if alert_type == "impossible_travel":
            return "false_positive"

        if self._state.difficulty == "easy":
            return "block_if_malicious"
        if self._state.difficulty == "medium":
            return "escalate_tier2"
        return "false_positive"

    def reset(self, seed=None, episode_id=None, **kwargs) -> SocObservation:
        # Accept difficulty from both direct and nested reset payloads.
        difficulty = kwargs.get("difficulty")
        mode = kwargs.get("mode")
        if difficulty is None and isinstance(kwargs.get("options"), dict):
            difficulty = kwargs["options"].get("difficulty")
            mode = mode or kwargs["options"].get("mode")
        if difficulty is None and isinstance(kwargs.get("config"), dict):
            difficulty = kwargs["config"].get("difficulty")
            mode = mode or kwargs["config"].get("mode")
        difficulty = difficulty or "easy"
        mode = mode or "single_agent"
        
        # Reset state completely
        self._state = SocState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            difficulty=difficulty,
            mode=mode,
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

        if self._state.mode == "campaign":
            self._state.remaining_steps = 30
            self._state.campaign_stages = ["recon", "foothold", "escalation", "persistence"]
            self._state.campaign_stage_index = 0
            self._state.episode_metrics.campaign_stage = self._state.campaign_stages[0]
            self._state.episode_metrics.campaign_progress = 0.0

        return SocObservation(
            done=False,
            reward=0.0,
            message=f"New alert received on difficulty: {difficulty}. Investigate and take_action.",
            remaining_steps=self._state.remaining_steps,
            alert_details=self._state.alert,
            evidence_collected=self._state.evidence_collected,
            score=self._state.score,
            mode=self._state.mode,
            active_agent=self._state.active_agent if self._state.mode != "single_agent" else None,
            transcript=self._state.transcript,
            agent_observations=self._build_agent_observations() if self._state.mode != "single_agent" else [],
            episode_metrics=self._state.episode_metrics,
        )

    def step(self, action: SocAction, timeout_s=None, **kwargs) -> SocObservation:
        if self._state.mode != "single_agent":
            return self._step_multi_agent(action)

        self._state.step_count += 1
        self._state.remaining_steps -= 1
        
        at = action.action_type
        reward = 0.0
        message = ""
        done = False

        if at == "search_logs":
            splunk_client = get_splunk_client()
            if splunk_client is not None:
                results = splunk_client.search(action.query or "")
                message = str(results)[:500]
                reward = 0.1
                if "logs_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("logs_checked")
            elif search_uploaded_logs(action.query or "", max_results=5):
                results = search_uploaded_logs(action.query or "", max_results=5)
                message = f"Uploaded log hits: {str(results)[:500]}"
                reward = 0.1
                if "logs_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("logs_checked")
            elif action.query == "203.0.113.5" and self._state.difficulty == "easy":
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
            expected = self._resolve_expected_decision()
            self._state.expected_decision = expected
            if action.decision == expected:
                if self._state.difficulty == "hard" and "ti_checked" not in self._state.evidence_collected:
                    message = f"You guessed {action.decision} but didn't check threat intel to verify! Score: 0.1."
                    reward = 0.1
                    self._state.score = 0.1
                else:
                    message = f"Correctly resolved alert as {action.decision}."
                    reward = 0.99
                    self._state.score = 0.99
            else:
                message = f"Incorrect decision: {action.decision}. Expected: {expected}."
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
            reward=reward,
            mode=self._state.mode,
            transcript=self._state.transcript,
            episode_metrics=self._state.episode_metrics,
        )

    def _build_agent_observations(self) -> List[AgentObservation]:
        transcript = self._state.transcript
        supervisor_msgs = [m for m in transcript if m.recipient in ("supervisor", "broadcast")]
        log_msgs = [m for m in transcript if m.recipient in ("log_hunter", "broadcast")]
        ti_msgs = [m for m in transcript if m.recipient in ("threat_intel", "broadcast")]

        return [
            AgentObservation(
                role="supervisor",
                visible_alert=self._state.alert,
                visible_evidence=self._state.evidence_collected,
                messages_for_agent=supervisor_msgs,
                step_hint="Coordinate specialists and decide final action.",
            ),
            AgentObservation(
                role="log_hunter",
                visible_alert={k: v for k, v in self._state.alert.items() if k in ("id", "type", "ip", "user")},
                visible_evidence=[e for e in self._state.evidence_collected if "log" in e],
                messages_for_agent=log_msgs,
                step_hint="Collect log evidence and report confidence.",
            ),
            AgentObservation(
                role="threat_intel",
                visible_alert={k: v for k, v in self._state.alert.items() if k in ("id", "type", "ip", "hash")},
                visible_evidence=[e for e in self._state.evidence_collected if "ti" in e],
                messages_for_agent=ti_msgs,
                step_hint="Enrich IoCs and report confidence.",
            ),
        ]

    def _append_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        payload: str,
        confidence: float = 0.5,
        evidence_refs: List[str] | None = None,
    ) -> None:
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            confidence=max(0.0, min(confidence, 1.0)),
            evidence_refs=evidence_refs or [],
        )
        self._state.transcript.append(msg)

    def _tick_turn(self) -> None:
        self._state.turn_index = (self._state.turn_index + 1) % len(self.TURN_ORDER)
        self._state.active_agent = self.TURN_ORDER[self._state.turn_index]

    def _evidence_sufficiency(self) -> float:
        required = {"logs_checked", "ti_checked"}
        present = set(self._state.evidence_collected)
        return len(required.intersection(present)) / len(required)

    def _coordination_efficiency(self) -> float:
        total = max(1, self._state.episode_metrics.total_actions)
        return self._state.episode_metrics.useful_actions / total

    def _advance_campaign_stage_if_needed(self, role: str, action_type: str) -> bool:
        if self._state.mode != "campaign" or not self._state.campaign_stages:
            return False
        current = self._state.campaign_stages[self._state.campaign_stage_index]
        transition_rules = {
            "recon": role == "log_hunter" and action_type in ("search_logs", "submit_log_report"),
            "foothold": role == "threat_intel" and action_type in ("get_threat_intel", "submit_ti_report"),
            "escalation": role == "supervisor" and action_type == "request_clarification",
            "persistence": role == "supervisor" and action_type == "take_action",
        }
        should_advance = transition_rules.get(current, False)
        if should_advance:
            stage_evidence = f"stage_{current}_complete"
            if stage_evidence not in self._state.evidence_collected:
                self._state.evidence_collected.append(stage_evidence)
            if self._state.campaign_stage_index < len(self._state.campaign_stages) - 1:
                self._state.campaign_stage_index += 1
            self._state.episode_metrics.campaign_stage = self._state.campaign_stages[self._state.campaign_stage_index]
            self._state.episode_metrics.campaign_progress = self._state.campaign_stage_index / (
                len(self._state.campaign_stages) - 1
            )
            return True
        return False

    def _memory_consistency_score(self) -> float:
        if self._state.mode != "campaign":
            return 0.0
        memories = self._state.per_agent_memory
        total_entries = sum(len(v) for v in memories.values())
        if total_entries == 0:
            return 0.0
        referenced = 0
        evidence_blob = " ".join(self._state.evidence_collected + [m.payload for m in self._state.transcript[-10:]])
        for entries in memories.values():
            for entry in entries[-4:]:
                key_terms = [term for term in entry.split() if len(term) > 4]
                if any(term in evidence_blob for term in key_terms):
                    referenced += 1
        return min(1.0, referenced / max(1, total_entries))

    def _step_multi_agent(self, action: SocAction) -> SocObservation:
        self._state.step_count += 1
        self._state.remaining_steps -= 1
        self._state.episode_metrics.total_actions += 1

        role = action.agent_role or self._state.active_agent
        reward = 0.0
        done = False
        message = ""

        if role != self._state.active_agent:
            self._state.episode_metrics.invalid_actions += 1
            self._state.episode_metrics.mistakes_made += 1
            self._state.mistake_made = True
            reward = -0.05
            message = f"Invalid turn. Active agent is {self._state.active_agent}, received action from {role}."
        elif role == "supervisor":
            if action.action_type == "delegate_log_hunter":
                self._append_message(
                    sender="supervisor",
                    recipient="log_hunter",
                    message_type="delegate",
                    payload=action.reason or "Investigate logs for this alert.",
                    confidence=action.confidence or 0.7,
                )
                reward = 0.05
                self._state.episode_metrics.useful_actions += 1
                message = "Supervisor delegated investigation to LogHunter."
            elif action.action_type == "delegate_threat_intel":
                self._append_message(
                    sender="supervisor",
                    recipient="threat_intel",
                    message_type="delegate",
                    payload=action.reason or "Perform threat intel enrichment.",
                    confidence=action.confidence or 0.7,
                )
                reward = 0.05
                self._state.episode_metrics.useful_actions += 1
                message = "Supervisor delegated enrichment to ThreatIntel."
            elif action.action_type == "request_clarification":
                self._append_message(
                    sender="supervisor",
                    recipient="broadcast",
                    message_type="clarification",
                    payload=action.reason or "Need confidence-adjusted follow-up evidence.",
                    confidence=action.confidence or 0.5,
                )
                self._state.episode_metrics.negotiation_rounds += 1
                reward = 0.02
                message = "Supervisor requested clarification from specialists."
            elif action.action_type == "take_action":
                expected = self._resolve_expected_decision()
                self._state.expected_decision = expected
                self._append_message(
                    sender="supervisor",
                    recipient="broadcast",
                    message_type="final_decision",
                    payload=f"Decision={action.decision}; reason={action.reason or 'n/a'}",
                    confidence=action.confidence or 0.6,
                    evidence_refs=self._state.evidence_collected,
                )
                done = True
                if action.decision == expected:
                    suff = self._evidence_sufficiency()
                    reward = 0.8 + (0.2 * suff)
                    self._state.score = min(0.99, reward)
                    self._state.episode_metrics.useful_actions += 1
                    message = f"Correctly resolved alert as {action.decision} with evidence sufficiency {suff:.2f}."
                else:
                    reward = -0.4
                    self._state.episode_metrics.mistakes_made += 1
                    self._state.mistake_made = True
                    self._state.score = 0.01
                    message = f"Incorrect decision: {action.decision}. Expected: {expected}."
            else:
                self._state.episode_metrics.invalid_actions += 1
                reward = -0.05
                message = f"Unsupported supervisor action: {action.action_type}."

        elif role == "log_hunter":
            if action.action_type == "search_logs":
                splunk_client = get_splunk_client()
                try:
                    if splunk_client is not None:
                        results = splunk_client.search(action.query or "")
                        report = str(results)[:300]
                    elif search_uploaded_logs(action.query or "", max_results=5):
                        results = search_uploaded_logs(action.query or "", max_results=5)
                        report = f"Uploaded log hits: {str(results)[:300]}"
                    else:
                        report = f"Mock logs for query={action.query or 'n/a'}"
                    if "logs_checked" not in self._state.evidence_collected:
                        self._state.evidence_collected.append("logs_checked")
                    self._append_message(
                        sender="log_hunter",
                        recipient="supervisor",
                        message_type="report",
                        payload=report,
                        confidence=action.confidence or 0.75,
                        evidence_refs=["logs_checked"],
                    )
                    self._state.episode_metrics.useful_actions += 1
                    self._state.episode_metrics.per_agent_rewards["log_hunter"] += 0.15
                    reward = 0.15
                    message = "LogHunter submitted a logs report."
                except Exception:
                    self._state.episode_metrics.tool_failures += 1
                    self._state.episode_metrics.mistakes_made += 1
                    self._state.mistake_made = True
                    reward = -0.1
                    message = "LogHunter failed to fetch logs."
            elif action.action_type == "submit_log_report":
                report = action.report or "Manual log report from analyst."
                self._append_message(
                    sender="log_hunter",
                    recipient="supervisor",
                    message_type="report",
                    payload=report,
                    confidence=action.confidence or 0.6,
                    evidence_refs=["logs_checked"] if "logs_checked" in self._state.evidence_collected else [],
                )
                reward = 0.08
                self._state.episode_metrics.useful_actions += 1
                message = "LogHunter sent a structured report."
            else:
                self._state.episode_metrics.invalid_actions += 1
                reward = -0.05
                message = f"Unsupported LogHunter action: {action.action_type}."

        elif role == "threat_intel":
            if action.action_type == "get_threat_intel":
                indicator = action.indicator or self._state.alert.get("ip") or self._state.alert.get("hash") or "unknown"
                report = f"Threat intel lookup on {indicator}: confidence medium-high."
                if indicator in ("203.0.113.5", "Emotet", "198.51.100.2"):
                    report = f"Threat intel on {indicator}: relevant contextual finding."
                if "ti_checked" not in self._state.evidence_collected:
                    self._state.evidence_collected.append("ti_checked")
                self._append_message(
                    sender="threat_intel",
                    recipient="supervisor",
                    message_type="report",
                    payload=report,
                    confidence=action.confidence or 0.75,
                    evidence_refs=["ti_checked"],
                )
                self._state.episode_metrics.useful_actions += 1
                self._state.episode_metrics.per_agent_rewards["threat_intel"] += 0.15
                reward = 0.15
                message = "ThreatIntel submitted an enrichment report."
            elif action.action_type == "submit_ti_report":
                report = action.report or "Manual threat intel summary."
                self._append_message(
                    sender="threat_intel",
                    recipient="supervisor",
                    message_type="report",
                    payload=report,
                    confidence=action.confidence or 0.6,
                    evidence_refs=["ti_checked"] if "ti_checked" in self._state.evidence_collected else [],
                )
                reward = 0.08
                self._state.episode_metrics.useful_actions += 1
                message = "ThreatIntel sent a structured report."
            else:
                self._state.episode_metrics.invalid_actions += 1
                reward = -0.05
                message = f"Unsupported ThreatIntel action: {action.action_type}."

        if not done and self._state.remaining_steps <= 0:
            done = True
            reward = -0.1
            self._state.score = 0.01
            message = "Max steps reached in multi-agent mode without a final decision."

        stage_advanced = self._advance_campaign_stage_if_needed(role=role, action_type=action.action_type)
        if stage_advanced:
            reward += 0.12
            message += " Campaign stage advanced."

        if self._state.mode == "campaign" and action.action_type != "take_action":
            immediate = reward * 0.35
            delayed = reward - immediate
            self._state.delayed_reward_bank += delayed
            reward = immediate
        elif self._state.mode == "campaign" and action.action_type == "take_action":
            reward += self._state.delayed_reward_bank
            self._state.delayed_reward_bank = 0.0

        if self._state.mistake_made and reward > 0.1 and action.action_type in (
            "submit_log_report",
            "submit_ti_report",
            "request_clarification",
            "take_action",
        ):
            self._state.recovered_after_mistake = True
            self._state.episode_metrics.recovery_actions += 1
            reward += 0.08
            message += " Recovery credit applied."

        memory_note = f"{role}:{action.action_type}:{(action.reason or action.report or action.query or action.indicator or '')}".strip()
        self._state.per_agent_memory[role].append(memory_note)

        self._state.episode_metrics.evidence_sufficiency = self._evidence_sufficiency()
        self._state.episode_metrics.coordination_efficiency = self._coordination_efficiency()
        self._state.episode_metrics.delayed_reward_buffer = self._state.delayed_reward_bank
        self._state.episode_metrics.memory_consistency_score = self._memory_consistency_score()
        if self._state.episode_metrics.mistakes_made > 0:
            self._state.episode_metrics.recovery_after_mistake = min(
                1.0, self._state.episode_metrics.recovery_actions / self._state.episode_metrics.mistakes_made
            )
        if self._state.mode == "campaign" and self._state.campaign_stages:
            self._state.episode_metrics.campaign_stage = self._state.campaign_stages[self._state.campaign_stage_index]
            self._state.episode_metrics.campaign_progress = self._state.campaign_stage_index / (
                len(self._state.campaign_stages) - 1
            )
        self._state.episode_metrics.per_agent_rewards["supervisor"] += reward if role == "supervisor" else 0.0

        if not done:
            self._tick_turn()

        return SocObservation(
            message=message,
            remaining_steps=self._state.remaining_steps,
            alert_details=self._state.alert,
            evidence_collected=self._state.evidence_collected,
            score=self._state.score,
            done=done,
            reward=reward,
            mode=self._state.mode,
            active_agent=self._state.active_agent if not done else None,
            transcript=self._state.transcript,
            agent_observations=self._build_agent_observations(),
            episode_metrics=self._state.episode_metrics,
        )

    @property
    def state(self) -> SocState:
        return self._state
