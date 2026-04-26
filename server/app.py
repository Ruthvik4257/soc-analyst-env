import sys
import os
from typing import Dict

# Add both parent directory and server directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server import create_fastapi_app
from fastapi import BackgroundTasks, File, UploadFile
from fastapi.responses import FileResponse
from environment import SocAnalystEnvironment
from models import SocAction, SocObservation
from server.datasets import (
    add_logs_from_content,
    clear_uploaded_logs,
    search_uploaded_logs,
    uploaded_logs_summary,
)
from server.integrations import SplunkClient, SplunkConfig, set_splunk_client

try:
    from server.rl_trainer import TRAINING_STATUS, run_training_loop
    _trainer_import_error = ""
except Exception as exc:  # pragma: no cover - keeps API bootable when RL stack is unavailable
    _trainer_import_error = str(exc)

    class _FallbackStatus:
        running = False
        total_episodes = 0
        completed_episodes = 0
        last_reward = 0.0
        last_message = f"Trainer unavailable: {_trainer_import_error}"
        model_name = ""
        mode = "single_agent"
        run_seed = 42
        coordination_efficiency = 0.0
        evidence_sufficiency = 0.0
        recovery_after_mistake = 0.0
        memory_consistency_score = 0.0
        campaign_progress = 0.0
        delayed_reward_success_rate = 0.0
        per_agent_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        policy_mode = "single_policy"
        role_model_names = {"supervisor": "", "log_hunter": "", "threat_intel": ""}
        per_role_last_rewards = {"supervisor": 0.0, "log_hunter": 0.0, "threat_intel": 0.0}
        report_path = ""
        training_history = []
        finetune_on_uploaded_logs = False
        finetune_dataset_size = 0

    TRAINING_STATUS = _FallbackStatus()

    def run_training_loop(*_args, **_kwargs):
        raise RuntimeError(f"RL trainer import failed: {_trainer_import_error}")

app = create_fastapi_app(SocAnalystEnvironment, SocAction, SocObservation)
MULTI_AGENT_SESSIONS: Dict[str, SocAnalystEnvironment] = {}

TRAINING_PRESETS = {
    "run1_smoke": {
        "episodes": 2,
        "model_name": "distilgpt2",
        "learning_rate": 1e-5,
        "push_to_hub": False,
        "mode": "single_agent",
        "campaign_length": 20,
        "negotiation_rounds": 2,
        "seed": 42,
    },
    "run2_multi_agent": {
        "episodes": 5,
        "model_name": "distilgpt2",
        "learning_rate": 8e-6,
        "push_to_hub": False,
        "mode": "multi_agent",
        "campaign_length": 20,
        "negotiation_rounds": 2,
        "seed": 42,
    },
    "run3_campaign": {
        "episodes": 8,
        "model_name": "distilgpt2",
        "learning_rate": 5e-6,
        "push_to_hub": False,
        "mode": "campaign",
        "campaign_length": 30,
        "negotiation_rounds": 3,
        "seed": 42,
    },
    "run4_uploaded_dataset_finetune": {
        "episodes": 6,
        "model_name": "distilgpt2",
        "learning_rate": 5e-6,
        "push_to_hub": False,
        "mode": "single_agent",
        "campaign_length": 20,
        "negotiation_rounds": 2,
        "seed": 42,
        "finetune_on_uploaded_logs": True,
        "finetune_max_logs": 200,
    },
}


@app.get("/")
def read_root():
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    return FileResponse(frontend_path)


@app.post("/api/integrations/splunk")
def configure_splunk(config: SplunkConfig):
    try:
        client = SplunkClient(config)
        set_splunk_client(client)
        return {"ok": True, "message": "Splunk integration configured."}
    except Exception as exc:
        return {"ok": False, "message": f"Failed to configure Splunk: {exc}"}


@app.post("/api/datasets/logs/upload")
async def upload_logs(file: UploadFile = File(...)):
    content = await file.read()
    inserted = add_logs_from_content(file.filename or "uploaded.log", content)
    return {
        "ok": True,
        "message": f"Uploaded {inserted} log entries from {file.filename}.",
        "summary": uploaded_logs_summary(),
    }


@app.post("/api/datasets/logs/clear")
def clear_logs():
    clear_uploaded_logs()
    return {"ok": True, "message": "Uploaded logs cleared.", "summary": uploaded_logs_summary()}


@app.get("/api/datasets/logs/search")
def search_logs(query: str = "", max_results: int = 20):
    rows = search_uploaded_logs(query, max_results=max_results)
    return {"ok": True, "query": query, "count": len(rows), "rows": rows, "summary": uploaded_logs_summary()}


@app.get("/api/datasets/logs/summary")
def logs_summary():
    return {"ok": True, "summary": uploaded_logs_summary()}


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "trainer_available": _trainer_import_error == "",
        "trainer_error": _trainer_import_error or None,
        "uploaded_logs": uploaded_logs_summary().get("total_logs", 0),
    }


@app.get("/api/train/presets")
def train_presets():
    return {"ok": True, "presets": TRAINING_PRESETS}


@app.post("/api/train")
def start_training(payload: dict, background_tasks: BackgroundTasks):
    if _trainer_import_error:
        return {"ok": False, "message": f"Training unavailable: {_trainer_import_error}"}
    if TRAINING_STATUS.running:
        return {"ok": False, "message": "Training already running."}
    episodes = int(payload.get("episodes", 1))
    model_name = payload.get("model_name")
    learning_rate = payload.get("learning_rate")
    push_to_hub = bool(payload.get("push_to_hub", False))
    mode = payload.get("mode", "single_agent")
    campaign_length = int(payload.get("campaign_length", 20))
    negotiation_rounds = int(payload.get("negotiation_rounds", 2))
    seed = int(payload.get("seed", 42))
    finetune_on_uploaded_logs = bool(payload.get("finetune_on_uploaded_logs", False))
    finetune_max_logs = int(payload.get("finetune_max_logs", 200))
    background_tasks.add_task(
        run_training_loop,
        episodes,
        model_name,
        learning_rate,
        push_to_hub,
        mode,
        campaign_length,
        negotiation_rounds,
        seed,
        finetune_on_uploaded_logs,
        finetune_max_logs,
    )
    return {
        "ok": True,
        "message": f"Training started for {episodes} episodes.",
        "model_name": model_name or "distilgpt2",
        "mode": mode,
        "finetune_on_uploaded_logs": finetune_on_uploaded_logs,
        "finetune_max_logs": finetune_max_logs,
    }


@app.post("/api/train/finetune")
def start_finetune(payload: dict, background_tasks: BackgroundTasks):
    payload = dict(payload or {})
    payload["finetune_on_uploaded_logs"] = True
    if "mode" not in payload:
        payload["mode"] = "single_agent"
    return start_training(payload, background_tasks)


@app.post("/api/multi/reset")
def multi_reset(payload: dict):
    difficulty = payload.get("difficulty", "easy")
    mode = payload.get("mode", "multi_agent")
    env = SocAnalystEnvironment()
    obs = env.reset(difficulty=difficulty, mode=mode)
    episode_id = env.state.episode_id
    MULTI_AGENT_SESSIONS[episode_id] = env
    return {"episode_id": episode_id, "observation": obs.model_dump()}


@app.post("/api/multi/step")
def multi_step(payload: dict):
    episode_id = payload.get("episode_id")
    action_payload = payload.get("action", {})
    if not episode_id or episode_id not in MULTI_AGENT_SESSIONS:
        return {"ok": False, "message": "Unknown multi-agent episode_id"}
    env = MULTI_AGENT_SESSIONS[episode_id]
    action = SocAction(**action_payload)
    obs = env.step(action)
    return {"ok": True, "episode_id": episode_id, "reward": obs.reward, "observation": obs.model_dump()}


@app.get("/api/eval/metrics")
def eval_metrics(episode_id: str):
    env = MULTI_AGENT_SESSIONS.get(episode_id)
    if env is None:
        return {"ok": False, "message": "Unknown multi-agent episode_id"}
    metrics = env.state.episode_metrics.model_dump()
    return {"ok": True, "episode_id": episode_id, "mode": env.state.mode, "metrics": metrics}


@app.get("/api/train/status")
def training_status():
    return {
        "running": TRAINING_STATUS.running,
        "total_episodes": TRAINING_STATUS.total_episodes,
        "completed_episodes": TRAINING_STATUS.completed_episodes,
        "last_reward": TRAINING_STATUS.last_reward,
        "last_message": TRAINING_STATUS.last_message,
        "model_name": TRAINING_STATUS.model_name,
        "mode": TRAINING_STATUS.mode,
        "run_seed": TRAINING_STATUS.run_seed,
        "coordination_efficiency": TRAINING_STATUS.coordination_efficiency,
        "evidence_sufficiency": TRAINING_STATUS.evidence_sufficiency,
        "recovery_after_mistake": TRAINING_STATUS.recovery_after_mistake,
        "memory_consistency_score": TRAINING_STATUS.memory_consistency_score,
        "campaign_progress": TRAINING_STATUS.campaign_progress,
        "delayed_reward_success_rate": TRAINING_STATUS.delayed_reward_success_rate,
        "training_backend": TRAINING_STATUS.training_backend,
        "per_agent_rewards": TRAINING_STATUS.per_agent_rewards,
        "policy_mode": TRAINING_STATUS.policy_mode,
        "role_model_names": TRAINING_STATUS.role_model_names,
        "per_role_last_rewards": TRAINING_STATUS.per_role_last_rewards,
        "report_path": TRAINING_STATUS.report_path,
        "finetune_on_uploaded_logs": TRAINING_STATUS.finetune_on_uploaded_logs,
        "finetune_dataset_size": TRAINING_STATUS.finetune_dataset_size,
    }


@app.get("/api/eval/report")
def eval_report():
    return {
        "ok": True,
        "mode": TRAINING_STATUS.mode,
        "model_name": TRAINING_STATUS.model_name,
        "episodes": TRAINING_STATUS.total_episodes,
        "completed_episodes": TRAINING_STATUS.completed_episodes,
        "coordination_efficiency": TRAINING_STATUS.coordination_efficiency,
        "evidence_sufficiency": TRAINING_STATUS.evidence_sufficiency,
        "recovery_after_mistake": TRAINING_STATUS.recovery_after_mistake,
        "memory_consistency_score": TRAINING_STATUS.memory_consistency_score,
        "campaign_progress": TRAINING_STATUS.campaign_progress,
        "delayed_reward_success_rate": TRAINING_STATUS.delayed_reward_success_rate,
        "training_backend": TRAINING_STATUS.training_backend,
        "per_agent_rewards": TRAINING_STATUS.per_agent_rewards,
        "policy_mode": TRAINING_STATUS.policy_mode,
        "role_model_names": TRAINING_STATUS.role_model_names,
        "per_role_last_rewards": TRAINING_STATUS.per_role_last_rewards,
        "history": TRAINING_STATUS.training_history,
        "report_path": TRAINING_STATUS.report_path,
        "finetune_on_uploaded_logs": TRAINING_STATUS.finetune_on_uploaded_logs,
        "finetune_dataset_size": TRAINING_STATUS.finetune_dataset_size,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)

if __name__ == '__main__':
    main()


