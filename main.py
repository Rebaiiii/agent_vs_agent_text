"""Run one visible Agent vs Agent text match."""

import argparse
from pathlib import Path

from agents import LLMAgent, RandomAgent, RuleAgent
from env import AgentVsAgentEnv
from llm_client import make_openai_client


AGENT_TYPES = ("random", "rule", "llm")


def main():
    args = parse_args()
    env = AgentVsAgentEnv(max_steps=args.max_steps)
    agents = {
        "white": make_agent("white", args.white),
        "black": make_agent("black", args.black),
    }

    log_lines = []
    observations = env.reset()
    write_line(log_lines, f"Match: white={args.white} black={args.black} max_steps={args.max_steps}")
    write_line(log_lines, "Initial grid:")
    write_line(log_lines, env.render_to_string())
    write_line(log_lines, "")
    death_stats = new_death_stats()

    done = False
    while not done:
        actions = {
            agent: agents[agent].act(observations[agent])
            for agent in ("white", "black")
        }
        next_observations, rewards, done, info = env.step(actions)
        for agent in ("white", "black"):
            agents[agent].observe_result(observations[agent], actions[agent], rewards[agent], info["agent_infos"][agent])
            update_death_stats(death_stats, info["agent_infos"][agent])
        observations = next_observations

        write_line(log_lines, f"Step {env.current_step}")
        for agent in ("white", "black"):
            agent_info = info["agent_infos"][agent]
            write_line(log_lines, f"Agent: {agent_info['agent']}")
            write_line(log_lines, f"Action: {agent_info['action']}")
            write_line(log_lines, f"Reward: {rewards[agent]}")
            write_line(log_lines, f"Message: {agent_info['message']}")
            write_line(log_lines, f"Event: {agent_info['event_type']}")
            if agent_info["death"]:
                owner = agent_info["death_trap_owner"] or "unknown"
                write_line(
                    log_lines,
                    (
                        f"Death: {agent_info['death_agent']} died to {agent_info['death_trap_type']} "
                        f"owned by {owner}; timeout={agent_info['death_timeout_started']} steps"
                    ),
                )
                if agent_info["dropped_items"]:
                    write_line(log_lines, f"Dropped items: {agent_info['dropped_items']}")
            if agent_info["timeout_skip"]:
                write_line(
                    log_lines,
                    f"Timeout: {agent_info['agent']} skipped this step; remaining={agent_info['death_timeout_remaining']}",
                )
        write_line(log_lines, f"White inventory: {info['white_inventory']}")
        write_line(log_lines, f"Black inventory: {info['black_inventory']}")
        write_line(log_lines, env.render_to_string())
        write_line(log_lines, "")

    winner = env.winner if env.winner is not None else "No winner"
    write_line(log_lines, f"Winner: {winner}")
    write_line(log_lines, "")
    write_line(log_lines, "Death summary:")
    for agent in ("white", "black"):
        write_line(log_lines, f"{agent} deaths: {death_stats[agent]['total']}")
        write_line(log_lines, f"{agent} own-trap deaths: {death_stats[agent]['own_trap']}")
        write_line(log_lines, f"{agent} object-trap deaths: {death_stats[agent]['object']}")
        write_line(log_lines, f"{agent} door-trap deaths: {death_stats[agent]['door']}")
        write_line(log_lines, f"{agent} exit-door-trap deaths: {death_stats[agent]['exit_door']}")
        write_line(log_lines, f"{agent} gun deaths: {death_stats[agent]['gun']}")
        write_line(log_lines, f"{agent} timeout skipped steps: {death_stats[agent]['timeout_skips']}")
    write_log(args.log_file, log_lines)
    print(f"Log written to {args.log_file}")


def make_agent(name, agent_type):
    if agent_type == "random":
        return RandomAgent(name)
    if agent_type == "rule":
        return RuleAgent(name)
    if agent_type == "llm":
        return LLMAgent(name, client_func=make_openai_client())
    raise ValueError(f"Unknown agent type: {agent_type}")


def write_line(log_lines, line):
    print(line)
    log_lines.append(line)


def write_log(path, log_lines):
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n")


def new_death_stats():
    return {
        "white": {
            "total": 0,
            "own_trap": 0,
            "object": 0,
            "door": 0,
            "exit_door": 0,
            "gun": 0,
            "timeout_skips": 0,
        },
        "black": {
            "total": 0,
            "own_trap": 0,
            "object": 0,
            "door": 0,
            "exit_door": 0,
            "gun": 0,
            "timeout_skips": 0,
        },
    }


def update_death_stats(death_stats, info):
    if info.get("timeout_skip"):
        death_stats[info["agent"]]["timeout_skips"] += 1

    if not info["death"]:
        return

    agent = info["death_agent"]
    trap_type = info["death_trap_type"]
    death_stats[agent]["total"] += 1
    if trap_type in death_stats[agent]:
        death_stats[agent][trap_type] += 1
    if info["death_to_own_trap"]:
        death_stats[agent]["own_trap"] += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Run one Agent vs Agent match.")
    parser.add_argument("--white", choices=AGENT_TYPES, default="llm", help="Agent type for white.")
    parser.add_argument("--black", choices=AGENT_TYPES, default="rule", help="Agent type for black.")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps before draw.")
    parser.add_argument("--log-file", default="logs/main_match.log", help="Path to write the full step log.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
