from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import login
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

from models import SocAction
from server.datasets import search_uploaded_logs
from server.environment import SocAnalystEnvironment
from server.integrations import execute_spl


@dataclass
class TrainingStatus:
    running: bool = False
    total_episodes: int = 0
    completed_episodes: int = 0
    last_reward: float = 0.0
    last_message: str = ""
    model_name: str = ""
    mode: str = "single_agent"
    run_seed: int = 42
    per_agent_rewards: Dict[str, float] = None  # type: ignore[assignment]
    coordination_efficiency: float = 0.0
    evidence_sufficiency: float = 0.0
    recovery_after_mistake: float = 0.0
    memory_consistency_score: float = 0.0
    delayed_reward_success_rate: float = 0.0
    campaign_progress: float = 0.0
    report_path: str = ""
    training_history: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_agent_rewards is None:
            self.per_agent_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        if self.training_history is None:
            self.training_history = []


TRAINING_STATUS = TrainingStatus()


def _build_loghunter_prompt(alert_details: Dict[str, Any]) -> str:
    return (
        "You are LogHunterAgent. Generate a concise SPL query for this alert.\n"
        f"Alert: {alert_details}\n"
        "Return only the SPL query."
    )


def _build_supervisor_prompt(alert_details: Dict[str, Any], logs: List[Dict[str, Any]]) -> str:
    return (
        "You are SupervisorAgent. Decide final SOC action.\n"
        f"Alert: {alert_details}\n"
        f"Logs: {logs}\n"
        "Return one of: false_positive, escalate_tier2, block_if_malicious."
    )


def _resolve_logs_for_query(query: str) -> List[Dict[str, Any]]:
    logs = execute_spl(query)
    if logs and not (len(logs) == 1 and "error" in logs[0]):
        return logs
    uploaded = search_uploaded_logs(query, max_results=10)
    return uploaded if uploaded else logs


def _parse_decision(text: str, default: str = "escalate_tier2") -> str:
    cleaned = text.strip().lower()
    choices = ["false_positive", "escalate_tier2", "block_if_malicious"]
    for choice in choices:
        if choice in cleaned:
            return choice
    return default


def _parse_spl(text: str, fallback: str) -> str:
    candidate = text.splitlines()[-1].strip()
    if not candidate:
        return fallback
    return candidate if candidate.startswith("search ") else f"search {candidate}"


def _sample_text(trainer: PPOTrainer, tokenizer: AutoTokenizer, prompt: str, device: str, max_new_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
    query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = trainer.model.generate(query_tensors, max_new_tokens=max_new_tokens, do_sample=True)
    response_tensors = generated[:, query_tensors.shape[1] :]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return query_tensors, response_tensors, text


def _train_single_agent_episode(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    env: SocAnalystEnvironment,
    reset_obs: Any,
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    alert_details = reset_obs.alert_details
    loghunter_prompt = _build_loghunter_prompt(alert_details)
    query_tensors, response_tensors, generated_text = _sample_text(
        trainer, tokenizer, loghunter_prompt, device, max_new_tokens=40
    )
    spl_query = _parse_spl(generated_text, fallback=f"search index=main {alert_details.get('id', '')}")
    logs = _resolve_logs_for_query(spl_query)

    supervisor_prompt = _build_supervisor_prompt(alert_details, logs)
    _, _, supervisor_text = _sample_text(trainer, tokenizer, supervisor_prompt, device, max_new_tokens=20)
    decision = _parse_decision(supervisor_text, default=random.choice(["escalate_tier2", "block_if_malicious"]))

    final_obs = env.step(
        SocAction(action_type="take_action", decision=decision, reason="SupervisorAgent final decision")
    )
    reward_value = float(final_obs.reward)
    trainer.step([query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)])
    episode_metrics = {
        "coordination_efficiency": 0.0,
        "evidence_sufficiency": 0.0,
        "per_agent_rewards": {"supervisor": reward_value, "log_hunter": 0.0, "threat_intel": 0.0},
    }
    return reward_value, episode_metrics


def _train_multi_agent_episode(
    trainer: PPOTrainer,
    tokenizer: AutoTokenizer,
    env: SocAnalystEnvironment,
    reset_obs: Any,
    device: str,
    campaign_length: int,
    negotiation_rounds: int,
) -> Tuple[float, Dict[str, Any]]:
    alert_details = reset_obs.alert_details

    # Supervisor delegates to log hunter.
    env.step(
        SocAction(
            action_type="delegate_log_hunter",
            agent_role="supervisor",
            reason="Investigate suspicious login behavior from alert context.",
            confidence=0.72,
        )
    )
    log_query = alert_details.get("ip") or alert_details.get("user") or alert_details.get("id", "A-000")
    env.step(
        SocAction(
            action_type="search_logs",
            agent_role="log_hunter",
            query=str(log_query),
            confidence=0.74,
        )
    )

    # Supervisor delegates to threat intel.
    env.step(
        SocAction(
            action_type="delegate_threat_intel",
            agent_role="supervisor",
            reason="Enrich IoCs and validate reputation signals.",
            confidence=0.73,
        )
    )
    ti_indicator = alert_details.get("ip") or alert_details.get("hash") or alert_details.get("id", "unknown")
    env.step(
        SocAction(
            action_type="get_threat_intel",
            agent_role="threat_intel",
            indicator=str(ti_indicator),
            confidence=0.74,
        )
    )

    for idx in range(max(0, negotiation_rounds)):
        env.step(
            SocAction(
                action_type="request_clarification",
                agent_role="supervisor",
                reason=f"Clarification round {idx + 1}: provide confidence-adjusted evidence summary.",
                confidence=0.68,
            )
        )
        if idx % 2 == 0:
            env.step(
                SocAction(
                    action_type="submit_log_report",
                    agent_role="log_hunter",
                    report="LogHunter follow-up: no benign baseline anomalies detected.",
                    confidence=0.7,
                )
            )
        else:
            env.step(
                SocAction(
                    action_type="submit_ti_report",
                    agent_role="threat_intel",
                    report="ThreatIntel follow-up: indicator overlaps known threat activity.",
                    confidence=0.7,
                )
            )

    # Long-horizon campaign-lite padding: maintain session to max campaign_length turns.
    while env.state.step_count < max(1, campaign_length - 1) and env.state.remaining_steps > 1:
        active = env.state.active_agent
        if active == "supervisor":
            env.step(
                SocAction(
                    action_type="request_clarification",
                    agent_role="supervisor",
                    reason="Campaign continuity check.",
                    confidence=0.6,
                )
            )
        elif active == "log_hunter":
            env.step(
                SocAction(
                    action_type="submit_log_report",
                    agent_role="log_hunter",
                    report="Campaign memory checkpoint from logs.",
                    confidence=0.6,
                )
            )
        else:
            env.step(
                SocAction(
                    action_type="submit_ti_report",
                    agent_role="threat_intel",
                    report="Campaign memory checkpoint from threat intel.",
                    confidence=0.6,
                )
            )

    final_prompt = (
        "You are SupervisorAgent. Decide final SOC action from this multi-agent transcript.\n"
        f"Alert={alert_details}\n"
        f"Evidence={env.state.evidence_collected}\n"
        f"Recent transcript={env.state.transcript[-6:]}\n"
        "Return one of: false_positive, escalate_tier2, block_if_malicious."
    )
    query_tensors, response_tensors, decision_text = _sample_text(
        trainer, tokenizer, final_prompt, device, max_new_tokens=20
    )
    decision = _parse_decision(decision_text, default="escalate_tier2")
    final_obs = env.step(
        SocAction(
            action_type="take_action",
            agent_role="supervisor",
            decision=decision,
            reason="PPO supervisor final decision",
            confidence=0.7,
        )
    )

    reward_value = float(final_obs.reward)
    trainer.step([query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)])
    metrics = final_obs.episode_metrics.model_dump() if final_obs.episode_metrics else {}
    return reward_value, metrics


def run_training_loop(
    episodes: int,
    model_name: Optional[str] = None,
    learning_rate: Optional[float] = None,
    push_to_hub: bool = False,
    mode: str = "single_agent",
    campaign_length: int = 20,
    negotiation_rounds: int = 2,
    seed: int = 42,
) -> None:
    resolved_model_name = model_name or os.getenv("HF_MODEL_NAME", "distilgpt2")
    resolved_learning_rate = learning_rate or float(os.getenv("TRAIN_LR", "1e-5"))
    output_dir = Path(os.getenv("TRAIN_OUTPUT_DIR", "./artifacts/ppo-soc-model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    training_mode = mode if mode in ("single_agent", "multi_agent", "campaign") else "single_agent"
    rng = random.Random(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    TRAINING_STATUS.running = True
    TRAINING_STATUS.total_episodes = episodes
    TRAINING_STATUS.completed_episodes = 0
    TRAINING_STATUS.last_reward = 0.0
    TRAINING_STATUS.last_message = "Training initialized"
    TRAINING_STATUS.model_name = resolved_model_name
    TRAINING_STATUS.mode = training_mode
    TRAINING_STATUS.run_seed = seed
    TRAINING_STATUS.per_agent_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
    TRAINING_STATUS.coordination_efficiency = 0.0
    TRAINING_STATUS.evidence_sufficiency = 0.0
    TRAINING_STATUS.recovery_after_mistake = 0.0
    TRAINING_STATUS.memory_consistency_score = 0.0
    TRAINING_STATUS.delayed_reward_success_rate = 0.0
    TRAINING_STATUS.campaign_progress = 0.0
    TRAINING_STATUS.training_history = []

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(resolved_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model.to(device)

    ppo_config = PPOConfig(batch_size=1, mini_batch_size=1, learning_rate=resolved_learning_rate)
    trainer = PPOTrainer(config=ppo_config, model=policy_model, tokenizer=tokenizer)

    difficulties = ["easy", "medium", "hard"]

    try:
        aggregate_per_agent = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        aggregate_coordination = 0.0
        aggregate_evidence = 0.0
        aggregate_recovery = 0.0
        aggregate_memory = 0.0
        aggregate_campaign_progress = 0.0
        delayed_reward_positive_count = 0

        for ep in range(episodes):
            env = SocAnalystEnvironment()
            selected_difficulty = difficulties[ep % len(difficulties)]
            reset_obs = env.reset(difficulty=selected_difficulty, mode=training_mode)

            if training_mode == "single_agent":
                reward_value, episode_metrics = _train_single_agent_episode(
                    trainer=trainer, tokenizer=tokenizer, env=env, reset_obs=reset_obs, device=device
                )
            else:
                # campaign mode currently reuses multi-agent mechanics with extended campaign horizon.
                effective_campaign_length = campaign_length if training_mode == "campaign" else max(8, campaign_length // 2)
                reward_value, episode_metrics = _train_multi_agent_episode(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    env=env,
                    reset_obs=reset_obs,
                    device=device,
                    campaign_length=effective_campaign_length,
                    negotiation_rounds=negotiation_rounds,
                )

            TRAINING_STATUS.completed_episodes = ep + 1
            TRAINING_STATUS.last_reward = reward_value
            TRAINING_STATUS.last_message = f"Episode {ep + 1}/{episodes} complete in {training_mode} mode."

            metrics_per_agent = episode_metrics.get("per_agent_rewards", {})
            for agent in aggregate_per_agent:
                aggregate_per_agent[agent] += float(metrics_per_agent.get(agent, 0.0))
            aggregate_coordination += float(episode_metrics.get("coordination_efficiency", 0.0))
            aggregate_evidence += float(episode_metrics.get("evidence_sufficiency", 0.0))
            aggregate_recovery += float(episode_metrics.get("recovery_after_mistake", 0.0))
            aggregate_memory += float(episode_metrics.get("memory_consistency_score", 0.0))
            aggregate_campaign_progress += float(episode_metrics.get("campaign_progress", 0.0))
            if float(episode_metrics.get("delayed_reward_buffer", 0.0)) > 0:
                delayed_reward_positive_count += 1

            TRAINING_STATUS.training_history.append(
                {
                    "episode": ep + 1,
                    "difficulty": selected_difficulty,
                    "mode": training_mode,
                    "reward": reward_value,
                    "coordination_efficiency": float(episode_metrics.get("coordination_efficiency", 0.0)),
                    "evidence_sufficiency": float(episode_metrics.get("evidence_sufficiency", 0.0)),
                    "recovery_after_mistake": float(episode_metrics.get("recovery_after_mistake", 0.0)),
                    "memory_consistency_score": float(episode_metrics.get("memory_consistency_score", 0.0)),
                    "campaign_progress": float(episode_metrics.get("campaign_progress", 0.0)),
                    "per_agent_rewards": metrics_per_agent,
                    "seed": seed,
                    "random_probe": round(rng.random(), 4),
                }
            )

        denom = max(1, episodes)
        TRAINING_STATUS.per_agent_rewards = {
            agent: round(total / denom, 4) for agent, total in aggregate_per_agent.items()
        }
        TRAINING_STATUS.coordination_efficiency = round(aggregate_coordination / denom, 4)
        TRAINING_STATUS.evidence_sufficiency = round(aggregate_evidence / denom, 4)
        TRAINING_STATUS.recovery_after_mistake = round(aggregate_recovery / denom, 4)
        TRAINING_STATUS.memory_consistency_score = round(aggregate_memory / denom, 4)
        TRAINING_STATUS.campaign_progress = round(aggregate_campaign_progress / denom, 4)
        TRAINING_STATUS.delayed_reward_success_rate = round(delayed_reward_positive_count / denom, 4)

        trainer.model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        report = {
            "mode": training_mode,
            "model_name": resolved_model_name,
            "episodes": episodes,
            "seed": seed,
            "campaign_length": campaign_length,
            "negotiation_rounds": negotiation_rounds,
            "avg_last_reward": TRAINING_STATUS.last_reward,
            "coordination_efficiency": TRAINING_STATUS.coordination_efficiency,
            "evidence_sufficiency": TRAINING_STATUS.evidence_sufficiency,
            "recovery_after_mistake": TRAINING_STATUS.recovery_after_mistake,
            "memory_consistency_score": TRAINING_STATUS.memory_consistency_score,
            "campaign_progress": TRAINING_STATUS.campaign_progress,
            "delayed_reward_success_rate": TRAINING_STATUS.delayed_reward_success_rate,
            "per_agent_rewards": TRAINING_STATUS.per_agent_rewards,
            "history": TRAINING_STATUS.training_history,
        }
        report_path = output_dir / "training_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        TRAINING_STATUS.report_path = str(report_path)

        if push_to_hub:
            hf_token = os.getenv("HF_TOKEN")
            repo_id = os.getenv("HF_REPO_ID")
            if hf_token and repo_id:
                login(token=hf_token)
                trainer.model.push_to_hub(repo_id)
                tokenizer.push_to_hub(repo_id)
                TRAINING_STATUS.last_message = f"Training complete. Model and report pushed to {repo_id}."
            else:
                TRAINING_STATUS.last_message = (
                    "Training complete. Set HF_TOKEN and HF_REPO_ID to enable push-to-hub."
                )
        else:
            TRAINING_STATUS.last_message = f"Training complete. Model and report saved to {output_dir}."

    except Exception as exc:  # pragma: no cover - runtime errors reported to UI
        TRAINING_STATUS.last_message = f"Training failed: {exc}"
    finally:
        TRAINING_STATUS.running = False
