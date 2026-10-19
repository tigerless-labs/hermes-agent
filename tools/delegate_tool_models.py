"""Per-task model choice for delegate_task (Lara fork addition).

Off unless ``delegation.models`` lists models. Then each task may name one of them in ``model`` and its child runs on
it, on the delegation's own provider and endpoint; a task that names none runs on ``delegation.model``. A model off the
list is refused, so the parent model picks only among models the operator approved.
"""

from typing import Any, Optional

MODEL_FIELD = "model"


def model_choices(cfg: Optional[dict]) -> list[str]:
    return [name for name in (str(m).strip() for m in (cfg or {}).get("models") or []) if name]


def _named(task: dict) -> str:
    return str(task.get(MODEL_FIELD) or "").strip()


def child_model(task: dict, cfg: Optional[dict], default: Optional[str]) -> Optional[str]:
    """The model a task's child runs on: its listed choice, else the delegation's default."""
    chosen = _named(task)
    return chosen if chosen in model_choices(cfg) else default


def unlisted_model(task_list: list[dict], cfg: Optional[dict]) -> Optional[str]:
    """An error for the first task naming a model off ``delegation.models``; None when every choice is listed."""
    choices = model_choices(cfg)
    if not choices:
        return None
    for index, task in enumerate(task_list):
        chosen = _named(task)
        if chosen and chosen not in choices:
            return f"Task {index} asks for model {chosen!r}; choose one of {', '.join(choices)}, or leave model out."
    return None


def with_model_choice(parameters: dict[str, Any], cfg: Optional[dict]) -> dict[str, Any]:
    """delegate_task parameters with a per-task ``model`` enum of the listed models; unchanged when none are listed."""
    choices = model_choices(cfg)
    if not choices:
        return parameters
    default = str((cfg or {}).get("model") or "").strip() or "your own model"
    field = {"type": "string", "enum": choices,
             "description": f"Optional model this child runs on. Without it, the child runs on {default}."}
    tasks = parameters["properties"]["tasks"]
    items = tasks["items"]
    return {**parameters, "properties": {**parameters["properties"], "tasks": {
        **tasks, "items": {**items, "properties": {**items["properties"], MODEL_FIELD: field}}}}}
