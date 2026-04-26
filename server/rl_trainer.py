from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    TRL_BACKEND = "ppo"
except Exception:
    AutoModelForCausalLMWithValueHead = AutoModelForCausalLM  # type: ignore[assignment]
    PPOConfig = None  # type: ignore[assignment]
    PPOTrainer = None  # type: ignore[assignment]
    TRL_BACKEND = "fallback_no_ppo"

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
    training_backend: str = TRL_BACKEND
    policy_mode: str = "single_policy"
    role_model_names: Dict[str, str] = None  # type: ignore[assignment]
    per_role_last_rewards: Dict[str, float] = None  # type: ignore[assignment]
    report_path: str = ""
    training_history: List[Dict[str, Any]] = None  # type: ignore[assignment]
    finetune_on_uploaded_logs: bool = False
    finetune_dataset_size: int = 0

    def __post_init__(self) -> None:
        if self.per_agent_rewards is None:
            self.per_agent_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        if self.role_model_names is None:
            self.role_model_names = {"supervisor": "", "log_hunter": "", "threat_intel": ""}
        if self.per_role_last_rewards is None:
            self.per_role_last_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
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


def _sample_text(trainer: Any, tokenizer: AutoTokenizer, prompt: str, device: str, max_new_tokens: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
    query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    model_obj = trainer.model if hasattr(trainer, "model") else trainer
    generated = model_obj.generate(query_tensors, max_new_tokens=max_new_tokens, do_sample=True)
    response_tensors = generated[:, query_tensors.shape[1] :]
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return query_tensors, response_tensors, text


def _build_uploaded_log_prompt(log_entry: Dict[str, Any]) -> str:
    fields = log_entry.get("fields") or {}
    fields_json = json.dumps(fields, ensure_ascii=True)
    return (
        "You are a SOC triage RL policy.\n"
        "Analyze the log and output a single action with concise reason.\n"
        "Valid actions: false_positive, escalate_tier2, block_if_malicious.\n"
        f"Log source: {log_entry.get('source', 'uploaded')}\n"
        f"Raw log: {log_entry.get('raw', '')}\n"
        f"Parsed fields: {fields_json}\n"
        "Output format: <action> | <reason>"
    )


def _score_uploaded_log_response(log_entry: Dict[str, Any], response_text: str) -> Tuple[float, Dict[str, Any]]:
    text = response_text.lower()
    decision = _parse_decision(text, default="escalate_tier2")
    reward = 0.0

    if decision in ("escalate_tier2", "block_if_malicious", "false_positive"):
        reward += 0.4

    raw = str(log_entry.get("raw", "")).lower()
    fields = log_entry.get("fields") or {}
    evidence_tokens = []
    for value in fields.values():
        token = str(value).strip().lower()
        if token and token not in ("none", "null"):
            evidence_tokens.append(token)
    if raw:
        evidence_tokens.extend([tok for tok in raw.split() if len(tok) > 4][:8])

    token_hits = 0
    for tok in evidence_tokens[:12]:
        if tok in text:
            token_hits += 1
    reward += min(0.6, token_hits * 0.1)

    if "reason" in text or "|" in response_text:
        reward += 0.1

    reward = max(-0.2, min(1.2, reward))
    metrics = {
        "coordination_efficiency": 0.0,
        "evidence_sufficiency": min(1.0, token_hits / 6.0),
        "per_agent_rewards": {"supervisor": reward, "log_hunter": 0.0, "threat_intel": 0.0},
        "predicted_decision": decision,
        "token_hits": token_hits,
    }
    return reward, metrics


def _train_uploaded_log_episode(
    trainer: Any,
    tokenizer: AutoTokenizer,
    device: str,
    log_entry: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    prompt = _build_uploaded_log_prompt(log_entry)
    query_tensors, response_tensors, generated_text = _sample_text(
        trainer=trainer,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=40,
    )
    reward_value, metrics = _score_uploaded_log_response(log_entry, generated_text)
    if hasattr(trainer, "step"):
        trainer.step([query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)])
    return reward_value, metrics


def _init_role_trainers(
    model_name: str,
    learning_rate: float,
    tokenizer: AutoTokenizer,
    device: str,
) -> Dict[str, Any]:
    role_trainers: Dict[str, Any] = {}
    for role in ("supervisor", "log_hunter", "threat_intel"):
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        model.to(device)
        if PPOTrainer is not None and PPOConfig is not None:
            cfg = PPOConfig(batch_size=1, mini_batch_size=1, learning_rate=learning_rate)
            role_trainers[role] = PPOTrainer(config=cfg, model=model, tokenizer=tokenizer)
        else:
            role_trainers[role] = model
    return role_trainers


def _train_single_agent_episode(
    trainer: Any,
    tokenizer: AutoTokenizer,
    env: SocAnalystEnvironment,
    reset_obs: Any,
    device: str,
) -> Tuple[float, Dict[str, Any]]:
    alert_details = reset_obs.alert_details
    loghunter_prompt = _build_loghunter_prompt(alert_details)
    query_tensors = tokenizer(loghunter_prompt, return_tensors="pt").input_ids.to(device)
    generated = trainer.model.generate(query_tensors, max_new_tokens=40, do_sample=True) if hasattr(trainer, "model") else trainer.generate(query_tensors, max_new_tokens=40, do_sample=True)
    response_tensors = generated[:, query_tensors.shape[1] :]
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    spl_query = _parse_spl(generated_text, fallback=f"search index=main {alert_details.get('id', '')}")
    logs = _resolve_logs_for_query(spl_query)

    supervisor_prompt = _build_supervisor_prompt(alert_details, logs)
    sup_query_tensors = tokenizer(supervisor_prompt, return_tensors="pt").input_ids.to(device)
    sup_generated = trainer.model.generate(sup_query_tensors, max_new_tokens=20, do_sample=True) if hasattr(trainer, "model") else trainer.generate(sup_query_tensors, max_new_tokens=20, do_sample=True)
    supervisor_text = tokenizer.decode(sup_generated[0], skip_special_tokens=True)
    decision = _parse_decision(supervisor_text, default=random.choice(["escalate_tier2", "block_if_malicious"]))

    final_obs = env.step(
        SocAction(action_type="take_action", decision=decision, reason="SupervisorAgent final decision")
    )
    reward_value = float(final_obs.reward)
    if hasattr(trainer, "step"):
        trainer.step([query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)])
    episode_metrics = {
        "coordination_efficiency": 0.0,
        "evidence_sufficiency": 0.0,
        "per_agent_rewards": {"supervisor": reward_value, "log_hunter": 0.0, "threat_intel": 0.0},
    }
    return reward_value, episode_metrics


def _train_multi_agent_episode(
    role_trainers: Dict[str, Any],
    tokenizer: AutoTokenizer,
    env: SocAnalystEnvironment,
    reset_obs: Any,
    device: str,
    campaign_length: int,
    negotiation_rounds: int,
) -> Tuple[float, Dict[str, Any]]:
    alert_details = reset_obs.alert_details
    role_last_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}

    def generate_for_role(role: str, prompt: str, max_new_tokens: int = 32) -> Tuple[torch.Tensor, torch.Tensor, str]:
        role_obj = role_trainers[role]
        if hasattr(role_obj, "model"):
            return _sample_text(role_obj, tokenizer, prompt, device, max_new_tokens=max_new_tokens)
        query_tensors = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated = role_obj.generate(query_tensors, max_new_tokens=max_new_tokens, do_sample=True)
        response_tensors = generated[:, query_tensors.shape[1] :]
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return query_tensors, response_tensors, text

    def ppo_step_for_role(role: str, query_tensors: torch.Tensor, response_tensors: torch.Tensor, reward_value: float) -> None:
        role_obj = role_trainers[role]
        if hasattr(role_obj, "step"):
            role_obj.step([query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)])
        role_last_rewards[role] = reward_value

    # Supervisor delegates to log hunter using supervisor-specific policy.
    sup_delegate_prompt = (
        "You are SupervisorAgent. Delegate to log_hunter with a short reason.\n"
        f"Alert={alert_details}\nReturn one delegation sentence."
    )
    sup_q1, sup_r1, sup_delegate_text = generate_for_role("supervisor", sup_delegate_prompt, max_new_tokens=20)
    obs1 = env.step(
        SocAction(
            action_type="delegate_log_hunter",
            agent_role="supervisor",
            reason=sup_delegate_text.splitlines()[-1][:180] or "Investigate suspicious login behavior.",
            confidence=0.72,
        )
    )
    ppo_step_for_role("supervisor", sup_q1, sup_r1, float(obs1.reward))

    log_query = alert_details.get("ip") or alert_details.get("user") or alert_details.get("id", "A-000")
    log_prompt = (
        "You are LogHunterAgent. Generate a concise search query for logs.\n"
        f"Alert={alert_details}\nReturn query only."
    )
    log_q, log_r, log_text = generate_for_role("log_hunter", log_prompt, max_new_tokens=18)
    log_generated_query = (log_text.splitlines()[-1].strip() or str(log_query))[:140]
    obs2 = env.step(
        SocAction(
            action_type="search_logs",
            agent_role="log_hunter",
            query=log_generated_query,
            confidence=0.74,
        )
    )
    ppo_step_for_role("log_hunter", log_q, log_r, float(obs2.reward))

    # Supervisor delegates to threat intel with supervisor-specific policy.
    sup_delegate_ti_prompt = (
        "You are SupervisorAgent. Delegate to threat_intel with a short reason.\n"
        f"Alert={alert_details}\nReturn one delegation sentence."
    )
    sup_q2, sup_r2, sup_ti_text = generate_for_role("supervisor", sup_delegate_ti_prompt, max_new_tokens=20)
    obs3 = env.step(
        SocAction(
            action_type="delegate_threat_intel",
            agent_role="supervisor",
            reason=sup_ti_text.splitlines()[-1][:180] or "Enrich IoCs and validate threat reputation.",
            confidence=0.73,
        )
    )
    ppo_step_for_role("supervisor", sup_q2, sup_r2, float(obs3.reward))

    ti_indicator = alert_details.get("ip") or alert_details.get("hash") or alert_details.get("id", "unknown")
    ti_prompt = (
        "You are ThreatIntelAgent. Produce a compact indicator to enrich.\n"
        f"Alert={alert_details}\nReturn indicator only."
    )
    ti_q, ti_r, ti_text = generate_for_role("threat_intel", ti_prompt, max_new_tokens=16)
    ti_generated_indicator = (ti_text.splitlines()[-1].strip() or str(ti_indicator))[:120]
    obs4 = env.step(
        SocAction(
            action_type="get_threat_intel",
            agent_role="threat_intel",
            indicator=ti_generated_indicator,
            confidence=0.74,
        )
    )
    ppo_step_for_role("threat_intel", ti_q, ti_r, float(obs4.reward))

    for idx in range(max(0, negotiation_rounds)):
        sup_clar_prompt = (
            "You are SupervisorAgent. Request one clarification from specialists.\n"
            f"Alert={alert_details}\nReturn one concise clarification request."
        )
        sup_qc, sup_rc, sup_clar_text = generate_for_role("supervisor", sup_clar_prompt, max_new_tokens=18)
        obs_c = env.step(
            SocAction(
                action_type="request_clarification",
                agent_role="supervisor",
                reason=sup_clar_text.splitlines()[-1][:180]
                or f"Clarification round {idx + 1}: provide confidence-adjusted evidence summary.",
                confidence=0.68,
            )
        )
        ppo_step_for_role("supervisor", sup_qc, sup_rc, float(obs_c.reward))

        if idx % 2 == 0:
            lh_report_prompt = (
                "You are LogHunterAgent. Write a short follow-up report sentence from log review."
            )
            lh_qr, lh_rr, lh_report_text = generate_for_role("log_hunter", lh_report_prompt, max_new_tokens=20)
            obs_lh = env.step(
                SocAction(
                    action_type="submit_log_report",
                    agent_role="log_hunter",
                    report=lh_report_text.splitlines()[-1][:220]
                    or "LogHunter follow-up: no benign baseline anomalies detected.",
                    confidence=0.7,
                )
            )
            ppo_step_for_role("log_hunter", lh_qr, lh_rr, float(obs_lh.reward))
        else:
            ti_report_prompt = (
                "You are ThreatIntelAgent. Write a short follow-up enrichment report sentence."
            )
            ti_qr, ti_rr, ti_report_text = generate_for_role("threat_intel", ti_report_prompt, max_new_tokens=20)
            obs_ti = env.step(
                SocAction(
                    action_type="submit_ti_report",
                    agent_role="threat_intel",
                    report=ti_report_text.splitlines()[-1][:220]
                    or "ThreatIntel follow-up: indicator overlaps known threat activity.",
                    confidence=0.7,
                )
            )
            ppo_step_for_role("threat_intel", ti_qr, ti_rr, float(obs_ti.reward))

    # Long-horizon campaign-lite padding: maintain session to max campaign_length turns.
    while env.state.step_count < max(1, campaign_length - 1) and env.state.remaining_steps > 1:
        active = env.state.active_agent
        if active == "supervisor":
            sup_pad_prompt = "You are SupervisorAgent. Give one short campaign continuity clarification request."
            sp_q, sp_r, sp_text = generate_for_role("supervisor", sup_pad_prompt, max_new_tokens=16)
            obs_sp = env.step(
                SocAction(
                    action_type="request_clarification",
                    agent_role="supervisor",
                    reason=sp_text.splitlines()[-1][:180] or "Campaign continuity check.",
                    confidence=0.6,
                )
            )
            ppo_step_for_role("supervisor", sp_q, sp_r, float(obs_sp.reward))
        elif active == "log_hunter":
            lh_pad_prompt = "You are LogHunterAgent. Provide one short campaign memory checkpoint."
            lp_q, lp_r, lp_text = generate_for_role("log_hunter", lh_pad_prompt, max_new_tokens=16)
            obs_lp = env.step(
                SocAction(
                    action_type="submit_log_report",
                    agent_role="log_hunter",
                    report=lp_text.splitlines()[-1][:200] or "Campaign memory checkpoint from logs.",
                    confidence=0.6,
                )
            )
            ppo_step_for_role("log_hunter", lp_q, lp_r, float(obs_lp.reward))
        else:
            ti_pad_prompt = "You are ThreatIntelAgent. Provide one short campaign memory checkpoint."
            tp_q, tp_r, tp_text = generate_for_role("threat_intel", ti_pad_prompt, max_new_tokens=16)
            obs_tp = env.step(
                SocAction(
                    action_type="submit_ti_report",
                    agent_role="threat_intel",
                    report=tp_text.splitlines()[-1][:200] or "Campaign memory checkpoint from threat intel.",
                    confidence=0.6,
                )
            )
            ppo_step_for_role("threat_intel", tp_q, tp_r, float(obs_tp.reward))

    final_prompt = (
        "You are SupervisorAgent. Decide final SOC action from this multi-agent transcript.\n"
        f"Alert={alert_details}\n"
        f"Evidence={env.state.evidence_collected}\n"
        f"Recent transcript={env.state.transcript[-6:]}\n"
        "Return one of: false_positive, escalate_tier2, block_if_malicious."
    )
    query_tensors, response_tensors, decision_text = _sample_text(
        role_trainers["supervisor"], tokenizer, final_prompt, device, max_new_tokens=20
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
    if hasattr(role_trainers["supervisor"], "step"):
        role_trainers["supervisor"].step(
            [query_tensors[0]], [response_tensors[0]], [torch.tensor(reward_value, device=device)]
        )
    role_last_rewards["supervisor"] = reward_value
    metrics = final_obs.episode_metrics.model_dump() if final_obs.episode_metrics else {}
    metrics["per_role_last_rewards"] = role_last_rewards
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
    finetune_on_uploaded_logs: bool = False,
    finetune_max_logs: int = 200,
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
    TRAINING_STATUS.training_backend = TRL_BACKEND
    TRAINING_STATUS.policy_mode = "multi_policy_by_role" if training_mode in ("multi_agent", "campaign") else "single_policy"
    TRAINING_STATUS.run_seed = seed
    TRAINING_STATUS.per_agent_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
    TRAINING_STATUS.role_model_names = {"supervisor": resolved_model_name, "log_hunter": resolved_model_name, "threat_intel": resolved_model_name}
    TRAINING_STATUS.per_role_last_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
    TRAINING_STATUS.coordination_efficiency = 0.0
    TRAINING_STATUS.evidence_sufficiency = 0.0
    TRAINING_STATUS.recovery_after_mistake = 0.0
    TRAINING_STATUS.memory_consistency_score = 0.0
    TRAINING_STATUS.delayed_reward_success_rate = 0.0
    TRAINING_STATUS.campaign_progress = 0.0
    TRAINING_STATUS.training_history = []
    TRAINING_STATUS.finetune_on_uploaded_logs = finetune_on_uploaded_logs
    TRAINING_STATUS.finetune_dataset_size = 0

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(resolved_model_name)
    policy_model.to(device)
    if PPOTrainer is not None and PPOConfig is not None:
        ppo_config = PPOConfig(batch_size=1, mini_batch_size=1, learning_rate=resolved_learning_rate)
        trainer: Any = PPOTrainer(config=ppo_config, model=policy_model, tokenizer=tokenizer)
    else:
        trainer = policy_model
    role_trainers: Optional[Dict[str, PPOTrainer]] = None
    if training_mode in ("multi_agent", "campaign"):
        role_trainers = _init_role_trainers(
            model_name=resolved_model_name,
            learning_rate=resolved_learning_rate,
            tokenizer=tokenizer,
            device=device,
        )

    difficulties = ["easy", "medium", "hard"]
    uploaded_dataset = []
    if finetune_on_uploaded_logs:
        uploaded_dataset = search_uploaded_logs("", max_results=max(1, int(finetune_max_logs)))
        TRAINING_STATUS.finetune_dataset_size = len(uploaded_dataset)
        if not uploaded_dataset:
            raise ValueError("No uploaded logs found. Upload dataset logs first, then retry fine-tuning.")

    try:
        aggregate_per_agent = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        aggregate_coordination = 0.0
        aggregate_evidence = 0.0
        aggregate_recovery = 0.0
        aggregate_memory = 0.0
        aggregate_campaign_progress = 0.0
        delayed_reward_positive_count = 0

        for ep in range(episodes):
            selected_difficulty = difficulties[ep % len(difficulties)]
            if finetune_on_uploaded_logs:
                log_entry = uploaded_dataset[ep % len(uploaded_dataset)]
                reward_value, episode_metrics = _train_uploaded_log_episode(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    device=device,
                    log_entry=log_entry,
                )
            else:
                env = SocAnalystEnvironment()
                reset_obs = env.reset(difficulty=selected_difficulty, mode=training_mode)

                if training_mode == "single_agent":
                    reward_value, episode_metrics = _train_single_agent_episode(
                        trainer=trainer, tokenizer=tokenizer, env=env, reset_obs=reset_obs, device=device
                    )
                else:
                    # campaign mode currently reuses multi-agent mechanics with extended campaign horizon.
                    effective_campaign_length = campaign_length if training_mode == "campaign" else max(8, campaign_length // 2)
                    reward_value, episode_metrics = _train_multi_agent_episode(
                        role_trainers=role_trainers or {
                            "supervisor": trainer,
                            "log_hunter": trainer,
                            "threat_intel": trainer,
                        },
                        tokenizer=tokenizer,
                        env=env,
                        reset_obs=reset_obs,
                        device=device,
                        campaign_length=effective_campaign_length,
                        negotiation_rounds=negotiation_rounds,
                    )

            TRAINING_STATUS.completed_episodes = ep + 1
            TRAINING_STATUS.last_reward = reward_value
            progress_mode = "uploaded_dataset_finetune" if finetune_on_uploaded_logs else training_mode
            TRAINING_STATUS.last_message = f"Episode {ep + 1}/{episodes} complete in {progress_mode} mode."
            if "per_role_last_rewards" in episode_metrics:
                TRAINING_STATUS.per_role_last_rewards = episode_metrics["per_role_last_rewards"]

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
                    "finetune_on_uploaded_logs": finetune_on_uploaded_logs,
                    "reward": reward_value,
                    "coordination_efficiency": float(episode_metrics.get("coordination_efficiency", 0.0)),
                    "evidence_sufficiency": float(episode_metrics.get("evidence_sufficiency", 0.0)),
                    "recovery_after_mistake": float(episode_metrics.get("recovery_after_mistake", 0.0)),
                    "memory_consistency_score": float(episode_metrics.get("memory_consistency_score", 0.0)),
                    "campaign_progress": float(episode_metrics.get("campaign_progress", 0.0)),
                    "per_agent_rewards": metrics_per_agent,
                    "per_role_last_rewards": episode_metrics.get("per_role_last_rewards", {}),
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

        trainer_model = trainer.model if hasattr(trainer, "model") else trainer
        trainer_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        if role_trainers is not None:
            role_dir = output_dir / "role_policies"
            role_dir.mkdir(parents=True, exist_ok=True)
            for role, role_trainer in role_trainers.items():
                role_path = role_dir / role
                if hasattr(role_trainer, "model"):
                    role_trainer.model.save_pretrained(str(role_path))
                else:
                    role_trainer.save_pretrained(str(role_path))
        report = {
            "mode": training_mode,
            "finetune_on_uploaded_logs": finetune_on_uploaded_logs,
            "finetune_dataset_size": len(uploaded_dataset),
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
            "policy_mode": TRAINING_STATUS.policy_mode,
            "role_model_names": TRAINING_STATUS.role_model_names,
            "per_role_last_rewards": TRAINING_STATUS.per_role_last_rewards,
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
                trainer_model = trainer.model if hasattr(trainer, "model") else trainer
                trainer_model.push_to_hub(repo_id)
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
