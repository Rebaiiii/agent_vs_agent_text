# Agent vs Agent Text

A small simultaneous-step text/grid game where two agents compete on a 5x5 grid of rooms to collect hidden items, avoid object and door traps, and escape first.

The current version is intentionally simple and deterministic except for agent choices.

## Game Rules

- White starts at `(0, 0)`.
- Black starts at `(4, 4)`.
- The exit room is `(4, 0)`.
- The exit is a special exit door connected to the outside.
- Required items are `key`, `passport`, `money`, and `gun`.
- Every grid tile is a room with 2 or 3 searchable objects.
- Example room objects include `desk`, `cabinet`, `painting`, `bookshelf`, `safe`, `drawer`, `vase`, and `locker`.
- Fixed hidden item locations:
  - `key` inside the `desk` at `(1, 3)`
  - `passport` inside the `safe` at `(3, 1)`
  - `money` inside the `cabinet` at `(2, 2)`
  - `gun` inside the `safe` at `(0, 4)`
- Each agent starts with 3 traps.
- Traps can be placed on room objects, normal doors, or the exit door.
- If an agent searches an object trapped by the opponent, the trap triggers, is removed, the agent drops all carried items, dies, returns to start, and waits out a timeout.
- If an agent uses an opponent-trapped door, the trap triggers, movement fails, the agent drops all carried items, dies, returns to start, and waits out a timeout.
- If an agent uses an opponent-trapped exit door, the trap triggers, escape fails, the agent drops all carried items, dies, returns to start, and waits out a timeout.
- If an agent has the `gun` and is in the same room as the opponent, `shoot` kills the opponent, makes them drop all carried items, and sends them to timeout.
- Both agents choose actions at the same time. The environment resolves those actions in deterministic white-then-black order inside the same timestep.
- Each simultaneous step costs 1 timestep.
- The first agent to collect all items and use `use_exit_door` in the exit room wins.
- If 500 turns pass with no winner, the game is a draw by default.
- Rewards include a default step penalty so agents prefer efficient play.

## Actions

- `up`
- `down`
- `left`
- `right`
- `use_exit_door` in the exit room
- Object search actions generated from the current room, such as `search_desk` or `search_safe`
- Object trap actions generated from the current room, such as `trap_cabinet` or `trap_locker`
- Door trap actions generated from connected doors, such as `trap_door_up` or `trap_door_right`
- Door inspection actions generated from connected doors, such as `inspect_door_up`
- `trap_exit_door` in the exit room
- `shoot` when holding the `gun` in the same room as the opponent

## Observation And Memory

The environment observation only includes what the current agent can perceive:

- current room position
- objects in the current room
- connected doors and door statuses
- whether the current room has the exit door
- nearby tile information
- inventory and needed items
- opponent position only when visible
- current step, max steps, and turns left
- actions available in the current room
- recent same-room opponent events observed since this agent last acted

Agent memory lives in `agents.py`, not inside the environment. `BaseAgent` tracks:

- visited rooms
- searched objects by room
- known empty objects
- known item locations and collected items
- suspected enemy traps
- own placed traps
- doors used safely
- suspected trapped doors
- own door traps and own exit door traps
- door trap trigger events
- whether the exit door is suspected dangerous
- chokepoint doors worth trapping
- observed opponent events
- observed opponent items
- observed opponent traps
- opponent searched objects and failed searches
- opponent last seen action
- last seen opponent position
- action history
- failed actions
- important events

After each environment step, callers should update both agents:

```python
observations = env.reset()
actions = {
    "white": agents["white"].act(observations["white"]),
    "black": agents["black"].act(observations["black"]),
}
next_observations, rewards, done, info = env.step(actions)

for agent_name in ("white", "black"):
    agents[agent_name].observe_result(
        observations[agent_name],
        actions[agent_name],
        rewards[agent_name],
        info["agent_infos"][agent_name],
    )
```

The scripts keep `next_observations` returned by `step()` as the next timestep's observations.

The top-level `info` dictionary includes `agent_infos` with one info dictionary per agent. Each per-agent info dictionary includes event metadata such as `event_type`, `found_item`, `found_items`, `searched_object`, `trap_placed`, `trap_type`, `door`, `trap_owner`, `triggered_trap`, `death`, `death_timeout_started`, `death_timeout_remaining`, `timeout_skip`, `dropped_items`, `movement_success`, `escape_success`, `action_success`, positions before and after the action, and `current_step`.

When an agent triggers a trap or is shot, that agent drops all inventory in the death room, returns to its start room, and enters a 10-step death timeout. During timeout, the agent can only `wait`; waiting still advances `current_step`.

## Same-Room Observability

When both agents are in the same room, the non-acting agent can observe public parts of the acting agent's action. Observed events are delivered through `recent_observed_events` and then cleared from that agent's inbox.

Visible same-room events include object searches, item finds, failed searches, object trap placement, door trap placement, exit door trap placement, door movement, trap triggers, shooting, exit door use, and failed exit door use. Hidden information is only revealed when the action itself makes it visible.

## Files

- `env.py`: The `AgentVsAgentEnv` game environment and action definitions.
- `agents.py`: `RandomAgent`, `RuleAgent`, and API-ready `LLMAgent`.
- `llm_client.py`: Minimal OpenAI-compatible chat client using only the standard library.
- `main.py`: Runs one visible match.
- `evaluate.py`: Runs batches of games and prints win/draw statistics.
- `README.md`: Project overview and usage notes.

## Run One Match

From this directory:

```bash
python3 main.py
```

You will see the grid after each simultaneous step, both agents' chosen actions, rewards, messages, and both inventories. The same full step log is also written to `logs/main_match.log` by default.
The log includes a death summary for white and black, split by trap type, gun deaths, own-trap deaths, and timeout skipped steps.

By default, `main.py` uses `LLMAgent` for white. If `OPENAI_API_KEY` is still missing or set to the placeholder, the LLM agent falls back to random valid actions.

Choose agent types with `--white` and `--black`:

```bash
python3 main.py --white rule --black random
python3 main.py --white llm --black rule
python3 main.py --white random --black llm
```

Available agent types are `random`, `rule`, and `llm`.

Set max steps and log path:

```bash
python3 main.py --white rule --black random --max-steps 500 --log-file logs/rule_vs_random.log
```

Grid symbols:

- `W`: white agent
- `B`: black agent
- `X`: both agents on the same tile
- `E`: exit
- `T`: trap
- `.`: empty tile

Items are hidden inside objects, so the grid does not reveal `key`, `passport`, `money`, or `gun` positions.

## Run Evaluation

From this directory:

```bash
python3 evaluate.py
```

This runs 100 games for each matchup:

- `RandomAgent` vs `RandomAgent`
- `RuleAgent` vs `RandomAgent`
- `RandomAgent` vs `RuleAgent`
- `RuleAgent` vs `RuleAgent`

It prints white wins, black wins, draws, average steps, average turns to win, win rates, trap counts, trap success rates, wins blocked by exit door traps, death counts by agent/trap type, gun deaths, and timeout skipped steps.

Under the current rules, agents only trigger opponent traps, so own-trap deaths should normally be `0`.

Evaluation also defaults to 500 max steps per game. Override it with:

```bash
python3 evaluate.py --max-steps 500
```

LLM evaluation is disabled by default to avoid accidental API usage. To include LLM matchups, either use CLI flags:

```bash
python3 evaluate.py --llm --llm-games 5
```

or set exported environment variables on the same command:

```bash
EVALUATE_LLM=true LLM_EVAL_GAMES=5 python3 evaluate.py
```

You can also set these in `.env`:

```bash
EVALUATE_LLM=true
LLM_EVAL_GAMES=5
```

## LLMAgent

`LLMAgent` can call an OpenAI-compatible chat completions API through `llm_client.py`.

LLM API settings live in `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_BASE_URL=https://api.openai.com/v1
EVALUATE_LLM=false
LLM_EVAL_GAMES=5
```

The project includes a tiny standard-library `.env` loader in `config.py`. The real `.env` file is ignored by git; use `.env.example` as the shareable template.

The model should return JSON text like:

```json
{"action": "search_desk", "reason": "The desk has not been searched."}
```

The LLM must choose an action from the current observation's `available_actions`. If the JSON cannot be parsed or the action is invalid, `LLMAgent` falls back to a random available action.

## Future Plan

- Add memory evaluation.
- Add hidden furniture.
- Add different trap types.
- Compare random, rule-based, and LLM agents.
