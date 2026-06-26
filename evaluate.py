"""Evaluate agent matchups over many games."""

import argparse

from agents import LLMAgent, RandomAgent, RuleAgent
from config import get_llm_config
from env import AgentVsAgentEnv
from llm_client import make_openai_client
from memory import (
    create_match_summary,
    load_strategy_memory,
    new_match_memory,
    save_strategy_memory,
    strategy_notes,
    update_match_memory,
    update_strategy_memory,
)


def run_match(agent_white, agent_black, render=False, max_steps=500, strategy_memory_file="strategy_memory.json"):
    env = AgentVsAgentEnv(max_steps=max_steps)
    agents = {
        "white": agent_white,
        "black": agent_black,
    }
    for agent in agents.values():
        agent.reset_match_memory()
    match_memory = new_match_memory({
        "white": _agent_type_name(agent_white),
        "black": _agent_type_name(agent_black),
    })

    observations = env.reset()
    done = False

    if render:
        env.render()
        print()

    metrics = {
        "object_traps_placed": 0,
        "door_traps_placed": 0,
        "exit_door_traps_placed": 0,
        "object_traps_triggered": 0,
        "door_traps_triggered": 0,
        "exit_door_traps_triggered": 0,
        "wins_blocked_by_exit_door_traps": 0,
        "white_deaths": 0,
        "black_deaths": 0,
        "white_own_trap_deaths": 0,
        "black_own_trap_deaths": 0,
        "white_object_trap_deaths": 0,
        "black_object_trap_deaths": 0,
        "white_door_trap_deaths": 0,
        "black_door_trap_deaths": 0,
        "white_exit_door_trap_deaths": 0,
        "black_exit_door_trap_deaths": 0,
        "white_gun_deaths": 0,
        "black_gun_deaths": 0,
        "white_timeout_skips": 0,
        "black_timeout_skips": 0,
    }

    while not done:
        actions = {
            agent: agents[agent].act(observations[agent])
            for agent in ("white", "black")
        }
        next_observations, rewards, done, info = env.step(actions)
        for agent in ("white", "black"):
            agent_info = info["agent_infos"][agent]
            agents[agent].observe_result(observations[agent], actions[agent], rewards[agent], agent_info)
            _update_metrics(metrics, agent_info)
        update_match_memory(match_memory, info)
        observations = next_observations

        if render:
            print(f"Step {env.current_step}: {info['message']}")
            env.render()
            print()

    match_summary = create_match_summary(match_memory, env.winner, env.current_step)
    strategy_memory = load_strategy_memory(strategy_memory_file)
    strategy_memory = update_strategy_memory(strategy_memory, match_summary)
    save_strategy_memory(strategy_memory, strategy_memory_file)

    return {
        "winner": env.winner,
        "steps": env.current_step,
        "metrics": metrics,
        "match_summary": match_summary,
    }


def evaluate(agent_white_factory, agent_black_factory, n_games=100, max_steps=500, strategy_memory_file="strategy_memory.json"):
    white_wins = 0
    black_wins = 0
    draws = 0
    total_steps = 0
    winning_steps = []
    totals = {
        "object_traps_placed": 0,
        "door_traps_placed": 0,
        "exit_door_traps_placed": 0,
        "object_traps_triggered": 0,
        "door_traps_triggered": 0,
        "exit_door_traps_triggered": 0,
        "wins_blocked_by_exit_door_traps": 0,
        "white_deaths": 0,
        "black_deaths": 0,
        "white_own_trap_deaths": 0,
        "black_own_trap_deaths": 0,
        "white_object_trap_deaths": 0,
        "black_object_trap_deaths": 0,
        "white_door_trap_deaths": 0,
        "black_door_trap_deaths": 0,
        "white_exit_door_trap_deaths": 0,
        "black_exit_door_trap_deaths": 0,
        "white_gun_deaths": 0,
        "black_gun_deaths": 0,
        "white_timeout_skips": 0,
        "black_timeout_skips": 0,
    }

    for _ in range(n_games):
        result = run_match(
            agent_white_factory(),
            agent_black_factory(),
            max_steps=max_steps,
            strategy_memory_file=strategy_memory_file,
        )
        total_steps += result["steps"]
        for key, value in result["metrics"].items():
            totals[key] += value

        if result["winner"] == "white":
            white_wins += 1
            winning_steps.append(result["steps"])
        elif result["winner"] == "black":
            black_wins += 1
            winning_steps.append(result["steps"])
        else:
            draws += 1

    average_steps = total_steps / n_games
    average_turns_to_win = sum(winning_steps) / len(winning_steps) if winning_steps else 0
    white_win_rate = white_wins / n_games
    black_win_rate = black_wins / n_games
    object_trap_success_rate = _success_rate(totals["object_traps_triggered"], totals["object_traps_placed"])
    door_trap_success_rate = _success_rate(totals["door_traps_triggered"], totals["door_traps_placed"])
    exit_door_trap_success_rate = _success_rate(
        totals["exit_door_traps_triggered"],
        totals["exit_door_traps_placed"],
    )

    print(f"Games: {n_games}")
    print(f"White wins: {white_wins}")
    print(f"Black wins: {black_wins}")
    print(f"Draws: {draws}")
    print(f"Average steps: {average_steps:.2f}")
    print(f"Average turns to win: {average_turns_to_win:.2f}")
    print(f"White win rate: {white_win_rate:.2%}")
    print(f"Black win rate: {black_win_rate:.2%}")
    print(f"Object traps placed: {totals['object_traps_placed']}")
    print(f"Door traps placed: {totals['door_traps_placed']}")
    print(f"Exit door traps placed: {totals['exit_door_traps_placed']}")
    print(f"Object trap success rate: {object_trap_success_rate:.2%}")
    print(f"Door trap success rate: {door_trap_success_rate:.2%}")
    print(f"Exit door trap success rate: {exit_door_trap_success_rate:.2%}")
    print(f"Wins blocked by exit door traps: {totals['wins_blocked_by_exit_door_traps']}")
    print(f"White deaths: {totals['white_deaths']}")
    print(f"Black deaths: {totals['black_deaths']}")
    print(f"White own-trap deaths: {totals['white_own_trap_deaths']}")
    print(f"Black own-trap deaths: {totals['black_own_trap_deaths']}")
    print(f"White timeout skipped steps: {totals['white_timeout_skips']}")
    print(f"Black timeout skipped steps: {totals['black_timeout_skips']}")
    print(
        "White deaths by trap type: "
        f"object={totals['white_object_trap_deaths']} "
        f"door={totals['white_door_trap_deaths']} "
        f"exit_door={totals['white_exit_door_trap_deaths']}"
    )
    print(
        "Black deaths by trap type: "
        f"object={totals['black_object_trap_deaths']} "
        f"door={totals['black_door_trap_deaths']} "
        f"exit_door={totals['black_exit_door_trap_deaths']}"
    )
    print(f"White gun deaths: {totals['white_gun_deaths']}")
    print(f"Black gun deaths: {totals['black_gun_deaths']}")

    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "average_steps": average_steps,
        "average_turns_to_win": average_turns_to_win,
        "white_win_rate": white_win_rate,
        "black_win_rate": black_win_rate,
        "object_trap_success_rate": object_trap_success_rate,
        "door_trap_success_rate": door_trap_success_rate,
        "exit_door_trap_success_rate": exit_door_trap_success_rate,
        **totals,
    }


def _update_metrics(metrics, info):
    event_type = info["event_type"]
    trap_type = info.get("trap_type")

    if event_type == "trap_placed" and trap_type == "object":
        metrics["object_traps_placed"] += 1
    elif event_type == "door_trap_placed":
        metrics["door_traps_placed"] += 1
    elif event_type == "exit_door_trap_placed":
        metrics["exit_door_traps_placed"] += 1
    elif event_type == "triggered_trap" and trap_type == "object":
        metrics["object_traps_triggered"] += 1
    elif event_type == "door_trap_triggered":
        metrics["door_traps_triggered"] += 1
    elif event_type == "exit_door_trap_triggered":
        metrics["exit_door_traps_triggered"] += 1
        if info.get("had_all_items_before"):
            metrics["wins_blocked_by_exit_door_traps"] += 1

    if info.get("death"):
        agent = info["death_agent"]
        trap_type = info["death_trap_type"]
        metrics[f"{agent}_deaths"] += 1
        if info.get("death_to_own_trap"):
            metrics[f"{agent}_own_trap_deaths"] += 1
        if trap_type == "object":
            metrics[f"{agent}_object_trap_deaths"] += 1
        elif trap_type == "door":
            metrics[f"{agent}_door_trap_deaths"] += 1
        elif trap_type == "exit_door":
            metrics[f"{agent}_exit_door_trap_deaths"] += 1
        elif trap_type == "gun":
            metrics[f"{agent}_gun_deaths"] += 1

    if info.get("timeout_skip"):
        metrics[f"{info['agent']}_timeout_skips"] += 1


def _success_rate(triggered, placed):
    if placed == 0:
        return 0
    return triggered / placed


def _agent_type_name(agent):
    class_name = agent.__class__.__name__
    if class_name.endswith("Agent"):
        class_name = class_name[:-5]
    return class_name.lower()


def print_matchup(title, white_factory, black_factory, max_steps, strategy_memory_file):
    print("=" * 60)
    print(title)
    print("=" * 60)
    evaluate(white_factory, black_factory, max_steps=max_steps, strategy_memory_file=strategy_memory_file)
    print()


def print_matchup_with_games(title, white_factory, black_factory, n_games, max_steps, strategy_memory_file):
    print("=" * 60)
    print(title)
    print("=" * 60)
    evaluate(
        white_factory,
        black_factory,
        n_games=n_games,
        max_steps=max_steps,
        strategy_memory_file=strategy_memory_file,
    )
    print()


def make_llm_agent(name, llm_config, strategy_memory_file):
    current_strategy_memory = load_strategy_memory(strategy_memory_file)
    return LLMAgent(
        name,
        client_func=make_openai_client(llm_config),
        llm_config=llm_config,
        strategy_memory_notes=strategy_notes(current_strategy_memory),
    )


def main():
    args = parse_args()
    llm_config = get_llm_config()
    evaluate_llm = args.llm or llm_config["evaluate_llm"]
    llm_eval_games = args.llm_games if args.llm_games is not None else llm_config["llm_eval_games"]
    max_steps = args.max_steps

    print_matchup(
        "RandomAgent vs RandomAgent",
        lambda: RandomAgent("white"),
        lambda: RandomAgent("black"),
        max_steps=max_steps,
        strategy_memory_file=args.strategy_memory_file,
    )
    print_matchup(
        "RuleAgent vs RandomAgent",
        lambda: RuleAgent("white"),
        lambda: RandomAgent("black"),
        max_steps=max_steps,
        strategy_memory_file=args.strategy_memory_file,
    )
    print_matchup(
        "RandomAgent vs RuleAgent",
        lambda: RandomAgent("white"),
        lambda: RuleAgent("black"),
        max_steps=max_steps,
        strategy_memory_file=args.strategy_memory_file,
    )
    print_matchup(
        "RuleAgent vs RuleAgent",
        lambda: RuleAgent("white"),
        lambda: RuleAgent("black"),
        max_steps=max_steps,
        strategy_memory_file=args.strategy_memory_file,
    )

    if evaluate_llm:
        print_matchup_with_games(
            "LLMAgent vs RuleAgent",
            lambda: make_llm_agent("white", llm_config, args.strategy_memory_file),
            lambda: RuleAgent("black"),
            n_games=llm_eval_games,
            max_steps=max_steps,
            strategy_memory_file=args.strategy_memory_file,
        )
        print_matchup_with_games(
            "RuleAgent vs LLMAgent",
            lambda: RuleAgent("white"),
            lambda: make_llm_agent("black", llm_config, args.strategy_memory_file),
            n_games=llm_eval_games,
            max_steps=max_steps,
            strategy_memory_file=args.strategy_memory_file,
        )
    else:
        print("LLM evaluation disabled. Use `python3 evaluate.py --llm --llm-games 5` to include LLMAgent matchups.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Agent vs Agent matchups.")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Include LLMAgent matchups. This may use your configured API key.",
    )
    parser.add_argument(
        "--llm-games",
        type=int,
        default=None,
        help="Number of games for each LLMAgent matchup.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per game before draw.",
    )
    parser.add_argument(
        "--strategy-memory-file",
        default="strategy_memory.json",
        help="Path to read and update persistent strategy memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
