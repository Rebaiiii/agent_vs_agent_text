"""Match and strategy memory helpers."""

import json
from copy import deepcopy
from pathlib import Path


STRATEGY_MEMORY_FILE = "strategy_memory.json"
AGENTS = ("white", "black")
TRAP_TYPES = ("object", "door", "exit_door")


DEFAULT_STRATEGY_MEMORY = {
    "games_played": 0,
    "exit_door_traps_placed": 0,
    "exit_door_traps_triggered": 0,
    "object_traps_placed": 0,
    "object_traps_triggered": 0,
    "door_traps_placed": 0,
    "door_traps_triggered": 0,
    "white_wins": 0,
    "black_wins": 0,
    "draws": 0,
    "agent_type_results": {},
}


def new_match_memory(agent_types=None):
    return {
        "agent_types": dict(agent_types or {}),
        "steps": [],
        "items_found": {agent: [] for agent in AGENTS},
        "traps_placed": {agent: {trap_type: 0 for trap_type in TRAP_TYPES} for agent in AGENTS},
        "trap_successes": {agent: {trap_type: 0 for trap_type in TRAP_TYPES} for agent in AGENTS},
        "important_mistakes": [],
        "useful_strategies_observed": [],
    }


def update_match_memory(match_memory, step_info):
    step_record = {
        "step": step_info["current_step"],
        "actions": dict(step_info["actions"]),
        "events": {},
    }

    for agent in AGENTS:
        agent_info = step_info["agent_infos"][agent]
        step_record["events"][agent] = {
            "event_type": agent_info["event_type"],
            "message": agent_info["message"],
        }
        _record_agent_event(match_memory, agent_info)

    match_memory["steps"].append(step_record)


def create_match_summary(match_memory, winner, total_steps):
    trap_successes = {
        agent: sum(match_memory["trap_successes"][agent].values())
        for agent in AGENTS
    }
    exit_door_trap_successes = {
        agent: match_memory["trap_successes"][agent]["exit_door"]
        for agent in AGENTS
    }
    return {
        "winner": winner,
        "total_steps": total_steps,
        "agent_types": dict(match_memory["agent_types"]),
        "items_found_by_agent": _items_found_by_agent(match_memory),
        "traps_placed_by_agent": match_memory["traps_placed"],
        "trap_successes": trap_successes,
        "trap_successes_by_owner_and_type": match_memory["trap_successes"],
        "exit_door_trap_successes": exit_door_trap_successes,
        "important_mistakes": list(match_memory["important_mistakes"]),
        "useful_strategies_observed": list(match_memory["useful_strategies_observed"]),
    }


def load_strategy_memory(path=STRATEGY_MEMORY_FILE):
    memory_path = Path(path)
    if not memory_path.exists():
        return deepcopy(DEFAULT_STRATEGY_MEMORY)

    try:
        loaded = json.loads(memory_path.read_text())
    except json.JSONDecodeError:
        loaded = {}

    memory = deepcopy(DEFAULT_STRATEGY_MEMORY)
    for key, value in loaded.items():
        if key in memory and isinstance(value, int):
            memory[key] = value
        elif key == "agent_type_results" and isinstance(value, dict):
            memory[key] = value
    return memory


def save_strategy_memory(strategy_memory, path=STRATEGY_MEMORY_FILE):
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(strategy_memory, indent=2, sort_keys=False) + "\n")


def update_strategy_memory(strategy_memory, match_summary):
    merged_memory = deepcopy(DEFAULT_STRATEGY_MEMORY)
    merged_memory.update(strategy_memory)
    strategy_memory = merged_memory
    strategy_memory["games_played"] += 1

    winner = match_summary["winner"]
    if winner == "white":
        strategy_memory["white_wins"] += 1
    elif winner == "black":
        strategy_memory["black_wins"] += 1
    else:
        strategy_memory["draws"] += 1

    _update_agent_type_results(strategy_memory, match_summary)

    for agent in AGENTS:
        traps_placed = match_summary["traps_placed_by_agent"][agent]
        strategy_memory["object_traps_placed"] += traps_placed["object"]
        strategy_memory["door_traps_placed"] += traps_placed["door"]
        strategy_memory["exit_door_traps_placed"] += traps_placed["exit_door"]

        trap_successes = match_summary["exit_door_trap_successes"][agent]
        strategy_memory["exit_door_traps_triggered"] += trap_successes

    object_successes = _trap_successes_by_type(match_summary, "object")
    door_successes = _trap_successes_by_type(match_summary, "door")
    strategy_memory["object_traps_triggered"] += object_successes
    strategy_memory["door_traps_triggered"] += door_successes

    return strategy_memory


def strategy_notes(strategy_memory):
    if not strategy_memory or strategy_memory.get("games_played", 0) == 0:
        return ""

    games = strategy_memory["games_played"]
    white_rate = strategy_memory["white_wins"] / games
    black_rate = strategy_memory["black_wins"] / games
    draw_rate = strategy_memory["draws"] / games
    object_rate = _success_rate(
        strategy_memory["object_traps_triggered"],
        strategy_memory["object_traps_placed"],
    )
    door_rate = _success_rate(
        strategy_memory["door_traps_triggered"],
        strategy_memory["door_traps_placed"],
    )
    exit_rate = _success_rate(
        strategy_memory["exit_door_traps_triggered"],
        strategy_memory["exit_door_traps_placed"],
    )
    return (
        f"Across {games} previous games: "
        f"white win rate {white_rate:.0%}, black win rate {black_rate:.0%}, draw rate {draw_rate:.0%}. "
        f"Trap success rates: object {object_rate:.0%}, door {door_rate:.0%}, "
        f"exit door {exit_rate:.0%}."
    )


def _record_agent_event(match_memory, info):
    agent = info["agent"]
    event_type = info["event_type"]
    trap_type = info.get("trap_type")

    if info.get("found_items"):
        for item in info["found_items"]:
            match_memory["items_found"][agent].append(item)

    if info.get("trap_placed") and trap_type in TRAP_TYPES:
        match_memory["traps_placed"][agent][trap_type] += 1
        _remember_unique(
            match_memory["useful_strategies_observed"],
            f"{agent} placed an object trap." if trap_type == "object" else f"{agent} placed a {trap_type} trap.",
        )

    if info.get("triggered_trap") and trap_type in TRAP_TYPES:
        owner = info.get("trap_owner")
        if owner in AGENTS:
            match_memory["trap_successes"][owner][trap_type] += 1
        _remember_unique(
            match_memory["important_mistakes"],
            f"{agent} triggered a {trap_type} trap owned by {owner or 'unknown'}.",
        )

    if event_type in ("invalid_action", "failed_exit_door", "failed_shoot", "no_door", "invalid_object"):
        _remember_unique(
            match_memory["important_mistakes"],
            f"{agent}: {info['message']}",
        )

    if event_type == "escaped":
        _remember_unique(
            match_memory["useful_strategies_observed"],
            f"{agent} reached the exit with all required items.",
        )


def _items_found_by_agent(match_memory):
    return {
        agent: {
            item: match_memory["items_found"][agent].count(item)
            for item in sorted(set(match_memory["items_found"][agent]))
        }
        for agent in AGENTS
    }


def _trap_successes_by_type(match_summary, trap_type):
    return sum(
        match_summary["trap_successes_by_owner_and_type"][agent][trap_type]
        for agent in AGENTS
    )


def _success_rate(triggered, placed):
    if placed == 0:
        return 0
    return triggered / placed


def _update_agent_type_results(strategy_memory, match_summary):
    agent_type_results = strategy_memory.setdefault("agent_type_results", {})
    agent_types = match_summary.get("agent_types", {})
    winner = match_summary["winner"]

    for agent in AGENTS:
        agent_type = agent_types.get(agent)
        if not agent_type:
            continue
        result = agent_type_results.setdefault(
            agent_type,
            {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        )
        result["games"] += 1
        if winner is None:
            result["draws"] += 1
        elif winner == agent:
            result["wins"] += 1
        else:
            result["losses"] += 1


def _remember_unique(items, value):
    if value not in items:
        items.append(value)
