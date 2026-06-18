"""Turn-based text environment for a small Agent vs Agent text game."""

MOVEMENT_ACTIONS = ["up", "down", "left", "right"]
DIRECTION_DELTAS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
ACTIONS = MOVEMENT_ACTIONS + ["use_exit_door", "shoot"]
REQUIRED_ITEMS = ["key", "passport", "money", "gun"]
OBJECT_POOL = ["desk", "cabinet", "painting", "bookshelf", "safe", "drawer", "vase", "locker"]


def get_door_key(room_a, room_b):
    return tuple(sorted([tuple(room_a), tuple(room_b)]))


class AgentVsAgentEnv:
    def __init__(self, size=5, max_steps=500, death_timeout_steps=10):
        self.size = size
        self.max_steps = max_steps
        self.death_timeout_steps = death_timeout_steps
        self.exit_position = (4, 0)
        self.start_positions = {
            "white": (0, 0),
            "black": (4, 4),
        }
        self.rooms = self._build_rooms()
        self.doors = self._build_doors()
        self.fixed_items = {
            ((1, 3), "desk"): ["key"],
            ((3, 1), "safe"): ["passport"],
            ((2, 2), "cabinet"): ["money"],
            ((0, 4), "safe"): ["gun"],
        }
        self.reset()

    @property
    def turn(self):
        return "simultaneous"

    def reset(self):
        self.positions = dict(self.start_positions)
        self.items = {
            object_key: list(items)
            for object_key, items in self.fixed_items.items()
        }
        self.inventories = {
            "white": [],
            "black": [],
        }
        self.object_traps = {
            "white": set(),
            "black": set(),
        }
        self.door_traps = {
            "white": set(),
            "black": set(),
        }
        self.traps_left = {
            "white": 3,
            "black": 3,
        }
        self.event_inbox = {
            "white": [],
            "black": [],
        }
        self.dead_until_step = {
            "white": 0,
            "black": 0,
        }
        self.current_step = 0
        self.done = False
        self.winner = None
        return self.observe_all()

    def observe_all(self):
        return {
            "white": self.observe("white"),
            "black": self.observe("black"),
        }

    def observe(self, agent):
        opponent = self.other_agent(agent)
        position = self.positions[agent]
        opponent_position = self.positions[opponent]
        distance = abs(position[0] - opponent_position[0]) + abs(position[1] - opponent_position[1])
        opponent_visible = distance <= 1
        opponent_in_room = position == opponent_position
        inventory = list(self.inventories[agent])
        needed_items = [item for item in REQUIRED_ITEMS if item not in inventory]
        objects = list(self.rooms[position])
        connected_doors = self._connected_doors(position)
        doors = {
            direction: "unknown"
            for direction in connected_doors
        }
        has_exit_door = position == self.exit_position
        recent_observed_events = list(self.event_inbox[agent])
        self.event_inbox[agent].clear()
        death_timeout_remaining = self._death_timeout_remaining(agent)
        is_dead = death_timeout_remaining > 0

        return {
            "agent": agent,
            "position": position,
            "room_position": position,
            "current_room": position,
            "objects": objects,
            "objects_in_room": objects,
            "doors": doors,
            "connected_doors": list(connected_doors.keys()),
            "has_exit_door": has_exit_door,
            "exit_door_status": "unknown" if has_exit_door else None,
            "known_enemy_traps": [],
            "recent_observed_events": recent_observed_events,
            "inventory": inventory,
            "needed_items": needed_items,
            "has_all_items": len(needed_items) == 0,
            "exit_room": self.exit_position,
            "exit_position": self.exit_position,
            "at_exit": has_exit_door,
            "opponent_visible": opponent_visible,
            "opponent_in_room": opponent_in_room,
            "opponent_position": opponent_position if opponent_visible else None,
            "nearby": self._nearby_info(agent, opponent_visible),
            "traps_left": self.traps_left[agent],
            "is_dead": is_dead,
            "death_timeout_remaining": death_timeout_remaining,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "turns_left": self.max_steps - self.current_step,
            "available_actions": ["wait"] if is_dead else self._available_actions(position, agent),
        }

    def step(self, actions):
        if self.done:
            rewards = {"white": 0, "black": 0}
            info = {
                "event_type": "game_over",
                "message": "Game is already over.",
                "actions": actions,
                "agent_infos": {
                    "white": self._game_over_agent_info("white", actions),
                    "black": self._game_over_agent_info("black", actions),
                },
                "current_step": self.current_step,
                "winner": self.winner,
                "white_inventory": list(self.inventories["white"]),
                "black_inventory": list(self.inventories["black"]),
            }
            return self.observe_all(), rewards, True, info

        if isinstance(actions, str):
            raise TypeError("Simultaneous steps require actions like {'white': action, 'black': action}.")

        actions = {
            "white": actions.get("white", "wait"),
            "black": actions.get("black", "wait"),
        }
        agent_infos = {}
        rewards = {}
        self.dead_this_step = set()

        for agent in ("white", "black"):
            info, reward = self._apply_agent_action(agent, actions[agent])
            agent_infos[agent] = info
            rewards[agent] = reward
            if info["death"]:
                dead_agent = info["death_agent"]
                self.dead_this_step.add(dead_agent)
                self.dead_until_step[dead_agent] = self.current_step + self.death_timeout_steps + 1

        self.current_step += 1

        for info in agent_infos.values():
            if info["death"]:
                dead_agent = info["death_agent"]
                info["death_timeout_started"] = self.death_timeout_steps
                info["death_timeout_remaining"] = self._death_timeout_remaining(dead_agent)

        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            self.winner = None

        for info in agent_infos.values():
            info["current_step"] = self.current_step
            info["white_inventory"] = list(self.inventories["white"])
            info["black_inventory"] = list(self.inventories["black"])
            public_event = self._make_public_event(info)
            if public_event:
                self.add_observable_event(info["agent"], public_event)

        messages = [
            f"{agent}: {info['message']}"
            for agent, info in agent_infos.items()
        ]
        info = {
            "event_type": "draw" if self.done and self.winner is None else "simultaneous_step",
            "message": " | ".join(messages),
            "actions": actions,
            "agent_infos": agent_infos,
            "current_step": self.current_step,
            "winner": self.winner,
            "white_inventory": list(self.inventories["white"]),
            "black_inventory": list(self.inventories["black"]),
        }

        return self.observe_all(), rewards, self.done, info

    def _apply_agent_action(self, agent, action):
        if agent in getattr(self, "dead_this_step", set()) or self._is_dead(agent):
            return self._dead_agent_info(agent, action), -1

        reward = -1
        message = "Turn complete."
        event_type = "turn"
        found_item = None
        found_items = []
        triggered_trap = False
        searched_object = None
        trap_placed = None
        trap_type = None
        door = None
        door_status = None
        trap_owner = None
        action_success = True
        movement_success = None
        escape_success = None
        result = {}
        position_before = self.positions[agent]
        had_all_items_before = all(item in self.inventories[agent] for item in REQUIRED_ITEMS)

        available_actions = self._available_actions(self.positions[agent], agent)

        if action not in available_actions:
            reward += -5
            message = "Invalid action."
            event_type = "invalid_action"
            action_success = False
        elif action.startswith("inspect_door_"):
            direction = action.removeprefix("inspect_door_")
            result = self._inspect_door(agent, direction)
            reward += result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            action_success = result["action_success"]
            door = result["door"]
            door_status = result.get("door_status")
        elif action in MOVEMENT_ACTIONS:
            result = self._move(agent, action)
            move_reward = result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            action_success = result["action_success"]
            triggered_trap = result.get("triggered_trap", False)
            trap_type = result.get("trap_type")
            door = result.get("door")
            trap_owner = result.get("trap_owner")
            movement_success = result.get("movement_success")
            reward += move_reward
        elif action.startswith("search_"):
            object_name = action.removeprefix("search_")
            result = self._search(agent, object_name)
            search_reward = result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            found_item = result["found_item"]
            found_items = result.get("found_items", [])
            triggered_trap = result["triggered_trap"]
            searched_object = result["searched_object"]
            action_success = result["action_success"]
            trap_type = result.get("trap_type")
            trap_owner = result.get("trap_owner")
            reward += search_reward
        elif action.startswith("trap_door_"):
            direction = action.removeprefix("trap_door_")
            result = self._place_door_trap(agent, direction)
            reward += result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            trap_placed = result["trap_placed"]
            trap_type = result["trap_type"]
            door = result["door"]
            action_success = result["action_success"]
        elif action == "trap_exit_door":
            result = self._place_exit_door_trap(agent)
            reward += result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            trap_placed = result["trap_placed"]
            trap_type = result["trap_type"]
            door = result.get("door")
            action_success = result["action_success"]
        elif action == "use_exit_door":
            result = self._use_exit_door(agent)
            reward += result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            triggered_trap = result.get("triggered_trap", False)
            trap_type = result.get("trap_type")
            door = result.get("door")
            trap_owner = result.get("trap_owner")
            action_success = result["action_success"]
            escape_success = result.get("escape_success", False)
        elif action == "shoot":
            result = self._shoot(agent)
            reward += result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            action_success = result["action_success"]
        elif action.startswith("trap_"):
            object_name = action.removeprefix("trap_")
            result = self._place_trap(agent, object_name)
            trap_reward = result["reward"]
            message = result["message"]
            event_type = result["event_type"]
            trap_placed = result["trap_placed"]
            trap_type = result.get("trap_type")
            action_success = result["action_success"]
            reward += trap_reward

        position_after = self.positions[agent]
        death = triggered_trap
        if result.get("death"):
            death = True
        death_agent = result.get("death_agent") if result.get("death") else (agent if death else None)
        death_trap_owner = result.get("death_trap_owner") if result.get("death") else (trap_owner if death else None)
        death_trap_type = result.get("death_trap_type") if result.get("death") else (trap_type if death else None)
        death_to_own_trap = result.get("death_to_own_trap", bool(death and trap_owner == agent))
        death_timeout_started = 0

        info = {
            "agent": agent,
            "action": action,
            "message": message,
            "event_type": event_type,
            "found_item": found_item,
            "found_items": found_items,
            "triggered_trap": triggered_trap,
            "trap_type": trap_type,
            "door": door,
            "door_status": door_status,
            "trap_owner": trap_owner,
            "searched_object": searched_object,
            "trap_placed": trap_placed,
            "action_success": action_success,
            "movement_success": movement_success,
            "escape_success": escape_success,
            "death": death,
            "death_agent": death_agent,
            "death_trap_owner": death_trap_owner,
            "death_trap_type": death_trap_type,
            "death_to_own_trap": death_to_own_trap,
            "death_timeout_started": death_timeout_started,
            "death_timeout_remaining": self._death_timeout_remaining(agent),
            "timeout_skip": False,
            "position_before": position_before,
            "position_after": position_after,
            "had_all_items_before": had_all_items_before,
            "dropped_items": list(result.get("dropped_items", [])),
            "current_step": self.current_step + 1,
            "winner": self.winner,
            "white_inventory": list(self.inventories["white"]),
            "black_inventory": list(self.inventories["black"]),
        }
        return info, reward

    def _dead_agent_info(self, agent, action):
        position = self.positions[agent]
        remaining_before = self._death_timeout_remaining(agent)
        remaining_after = max(0, remaining_before - 1)
        message = (
            f"{agent} is out after triggering a trap. "
            f"Death timeout remaining: {remaining_after} steps."
        )
        event_type = "death_timeout"
        action_success = False

        return {
            "agent": agent,
            "action": action,
            "message": message,
            "event_type": event_type,
            "found_item": None,
            "found_items": [],
            "triggered_trap": False,
            "trap_type": None,
            "door": None,
            "door_status": None,
            "trap_owner": None,
            "searched_object": None,
            "trap_placed": None,
            "action_success": action_success,
            "movement_success": False,
            "escape_success": False,
            "death": False,
            "death_agent": None,
            "death_trap_owner": None,
            "death_trap_type": None,
            "death_to_own_trap": False,
            "death_timeout_started": 0,
            "death_timeout_remaining": remaining_after,
            "death_timeout_remaining_before": remaining_before,
            "timeout_skip": True,
            "position_before": position,
            "position_after": position,
            "had_all_items_before": all(item in self.inventories[agent] for item in REQUIRED_ITEMS),
            "dropped_items": [],
            "current_step": self.current_step + 1,
            "winner": self.winner,
            "white_inventory": list(self.inventories["white"]),
            "black_inventory": list(self.inventories["black"]),
        }

    def _game_over_agent_info(self, agent, actions):
        position = self.positions[agent]
        action = actions.get(agent, "wait") if isinstance(actions, dict) else actions
        return {
            "agent": agent,
            "action": action,
            "message": "Game is already over.",
            "event_type": "game_over",
            "found_item": None,
            "found_items": [],
            "triggered_trap": False,
            "trap_type": None,
            "door": None,
            "door_status": None,
            "trap_owner": None,
            "searched_object": None,
            "trap_placed": None,
            "action_success": False,
            "movement_success": False,
            "escape_success": False,
            "death": False,
            "death_agent": None,
            "death_trap_owner": None,
            "death_trap_type": None,
            "death_to_own_trap": False,
            "death_timeout_started": 0,
            "death_timeout_remaining": self._death_timeout_remaining(agent),
            "timeout_skip": False,
            "position_before": position,
            "position_after": position,
            "had_all_items_before": all(item in self.inventories[agent] for item in REQUIRED_ITEMS),
            "dropped_items": [],
            "current_step": self.current_step,
            "winner": self.winner,
            "white_inventory": list(self.inventories["white"]),
            "black_inventory": list(self.inventories["black"]),
        }

    def render(self):
        grid = self.render_to_string()
        print(grid)
        return grid

    def render_to_string(self):
        rows = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                position = (x, y)
                cell = "."
                if position == self.exit_position:
                    cell = "E"
                if self._room_has_trap(position):
                    cell = "T"
                if self.positions["white"] == position:
                    cell = "W"
                if self.positions["black"] == position:
                    cell = "B" if cell != "W" else "X"
                row.append(cell)
            rows.append(" ".join(row))
        return "\n".join(rows)

    def other_agent(self, agent):
        return "black" if agent == "white" else "white"

    def add_observable_event(self, actor, event):
        observer = self.other_agent(actor)
        if self.positions[observer] == tuple(event["room"]):
            self.event_inbox[observer].append(event)

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_dead(self, agent):
        return self._death_timeout_remaining(agent) > 0

    def _death_timeout_remaining(self, agent):
        return max(0, self.dead_until_step[agent] - self.current_step)

    def _nearby_info(self, agent, opponent_visible):
        nearby = {}
        x, y = self.positions[agent]
        opponent = self.other_agent(agent)
        opponent_position = self.positions[opponent]
        for direction, (dx, dy) in DIRECTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            position = (nx, ny)
            if not self._has_door((x, y), position):
                nearby[direction] = "no door"
            elif opponent_visible and position == opponent_position:
                nearby[direction] = "opponent"
            else:
                nearby[direction] = "unknown"
        return nearby

    def _move(self, agent, action):
        position = self.positions[agent]
        next_position = self._neighbor(position, action)

        if not self._has_door(position, next_position):
            return {
                "reward": -3,
                "message": f"No door {action}.",
                "event_type": "no_door",
                "action_success": False,
                "movement_success": False,
            }

        door_key = get_door_key(position, next_position)
        opponent = self.other_agent(agent)
        if door_key in self.door_traps[opponent]:
            self.door_traps[opponent].remove(door_key)
            dropped_items = self._kill_agent(agent, position)
            self.positions[agent] = self.start_positions[agent]
            return {
                "reward": -15,
                "message": f"Triggered opponent door trap while moving {action}.",
                "event_type": "door_trap_triggered",
                "action_success": False,
                "triggered_trap": True,
                "trap_type": "door",
                "door": door_key,
                "trap_owner": opponent,
                "movement_success": False,
                "dropped_items": dropped_items,
            }

        self.positions[agent] = next_position
        return {
            "reward": 0,
            "message": f"Moved {action}.",
            "event_type": "move",
            "action_success": True,
            "door": door_key,
            "movement_success": True,
        }

    def _search(self, agent, object_name):
        position = self.positions[agent]
        object_key = (position, object_name)
        opponent = self.other_agent(agent)

        if object_name not in self.rooms[position]:
            return {
                "reward": -5,
                "message": f"There is no {object_name} in this room.",
                "event_type": "invalid_object",
                "found_item": None,
                "triggered_trap": False,
                "searched_object": object_name,
                "action_success": False,
            }

        if object_key in self.object_traps[opponent]:
            self.object_traps[opponent].remove(object_key)
            dropped_items = self._kill_agent(agent, position)
            self.positions[agent] = self.start_positions[agent]
            return {
                "reward": -15,
                "message": f"Triggered opponent trap on {object_name} and returned to start.",
                "event_type": "triggered_trap",
                "found_item": None,
                "triggered_trap": True,
                "trap_type": "object",
                "trap_owner": opponent,
                "searched_object": object_name,
                "action_success": False,
                "dropped_items": dropped_items,
            }

        found_items = self.items.get(object_key, [])
        if not found_items:
            return {
                "reward": -1,
                "message": f"Searched {object_name} and found nothing.",
                "event_type": "empty_search",
                "found_item": None,
                "found_items": [],
                "triggered_trap": False,
                "searched_object": object_name,
                "action_success": True,
            }

        for item in found_items:
            if item not in self.inventories[agent]:
                self.inventories[agent].append(item)
        del self.items[object_key]
        item_text = ", ".join(found_items)
        return {
            "reward": 10,
            "message": f"Found {item_text} in {object_name}.",
            "event_type": "found_item",
            "found_item": found_items[0],
            "found_items": list(found_items),
            "triggered_trap": False,
            "searched_object": object_name,
            "action_success": True,
        }

    def _place_trap(self, agent, object_name):
        if self.traps_left[agent] <= 0:
            return {
                "reward": -3,
                "message": "No traps left.",
                "event_type": "no_traps_left",
                "trap_placed": None,
                "action_success": False,
            }

        position = self.positions[agent]
        if object_name not in self.rooms[position]:
            return {
                "reward": -5,
                "message": f"There is no {object_name} in this room.",
                "event_type": "invalid_object",
                "trap_placed": None,
                "action_success": False,
            }

        object_key = (position, object_name)
        if object_key in self.object_traps[agent]:
            return {
                "reward": -3,
                "message": f"{object_name} is already trapped.",
                "event_type": "trap_already_placed",
                "trap_placed": None,
                "trap_type": "object",
                "action_success": False,
            }

        self.object_traps[agent].add(object_key)
        self.traps_left[agent] -= 1
        return {
            "reward": -1,
            "message": f"Placed a trap on {object_name}.",
            "event_type": "trap_placed",
            "trap_placed": object_name,
            "trap_type": "object",
            "action_success": True,
        }

    def _place_door_trap(self, agent, direction):
        if self.traps_left[agent] <= 0:
            return self._trap_failure("No traps left.", "no_traps_left", "door")

        position = self.positions[agent]
        next_position = self._neighbor(position, direction)
        if not self._has_door(position, next_position):
            return self._trap_failure(f"No door {direction}.", "no_door", "door")

        door_key = get_door_key(position, next_position)
        if door_key in self.door_traps[agent]:
            return self._trap_failure("Door is already trapped.", "trap_already_placed", "door", door_key)

        self.door_traps[agent].add(door_key)
        self.traps_left[agent] -= 1
        return {
            "reward": -1,
            "message": f"Placed a trap on the {direction} door.",
            "event_type": "door_trap_placed",
            "trap_placed": direction,
            "trap_type": "door",
            "door": door_key,
            "action_success": True,
        }

    def _place_exit_door_trap(self, agent):
        if self.traps_left[agent] <= 0:
            return self._trap_failure("No traps left.", "no_traps_left", "exit_door")

        position = self.positions[agent]
        if position != self.exit_position:
            return self._trap_failure("This room has no exit door.", "no_exit_door", "exit_door")

        trap_key = self._exit_door_key()
        if trap_key in self.door_traps[agent]:
            return self._trap_failure("Exit door is already trapped.", "trap_already_placed", "exit_door")

        self.door_traps[agent].add(trap_key)
        self.traps_left[agent] -= 1
        return {
            "reward": -1,
            "message": "Placed a trap on the exit door.",
            "event_type": "exit_door_trap_placed",
            "trap_placed": "exit_door",
            "trap_type": "exit_door",
            "door": trap_key,
            "action_success": True,
        }

    def _inspect_door(self, agent, direction):
        position = self.positions[agent]
        next_position = self._neighbor(position, direction)
        if not self._has_door(position, next_position):
            return {
                "reward": -1,
                "message": f"No door {direction} to inspect.",
                "event_type": "inspect_no_door",
                "door": None,
                "action_success": False,
            }

        door_key = get_door_key(position, next_position)
        opponent = self.other_agent(agent)
        status = "suspected trapped" if door_key in self.door_traps[opponent] else "known safe"
        return {
            "reward": -1,
            "message": f"Inspected {direction} door: {status}.",
            "event_type": "door_inspected",
            "door": door_key,
            "door_status": status,
            "action_success": True,
        }

    def _use_exit_door(self, agent):
        position = self.positions[agent]
        if position != self.exit_position:
            return {
                "reward": -5,
                "message": "There is no exit door in this room.",
                "event_type": "failed_exit_door",
                "action_success": False,
                "escape_success": False,
            }

        opponent = self.other_agent(agent)
        trap_key = self._exit_door_key()
        if trap_key in self.door_traps[opponent]:
            self.door_traps[opponent].remove(trap_key)
            dropped_items = self._kill_agent(agent, position)
            self.positions[agent] = self.start_positions[agent]
            return {
                "reward": -15,
                "message": "Triggered opponent trap on the exit door.",
                "event_type": "exit_door_trap_triggered",
                "triggered_trap": True,
                "trap_type": "exit_door",
                "trap_owner": opponent,
                "door": trap_key,
                "action_success": False,
                "escape_success": False,
                "dropped_items": dropped_items,
            }

        has_all_items = all(item in self.inventories[agent] for item in REQUIRED_ITEMS)
        if has_all_items:
            self.done = True
            self.winner = agent
            return {
                "reward": 50,
                "message": f"{agent} used the exit door and won.",
                "event_type": "escaped",
                "action_success": True,
                "escape_success": True,
            }
        return {
            "reward": -5,
            "message": "Exit door use failed: missing required items.",
            "event_type": "failed_exit_door",
            "action_success": False,
            "escape_success": False,
        }

    def _shoot(self, agent):
        opponent = self.other_agent(agent)
        position = self.positions[agent]
        if "gun" not in self.inventories[agent]:
            return {
                "reward": -5,
                "message": "Cannot shoot without the gun.",
                "event_type": "failed_shoot",
                "action_success": False,
            }
        if self.positions[opponent] != position:
            return {
                "reward": -5,
                "message": "Cannot shoot because the opponent is not in this room.",
                "event_type": "failed_shoot",
                "action_success": False,
            }
        if self._is_dead(opponent):
            return {
                "reward": -5,
                "message": "Cannot shoot because the opponent is already out.",
                "event_type": "failed_shoot",
                "action_success": False,
            }

        dropped_items = self._kill_agent(opponent, position)
        self.positions[opponent] = self.start_positions[opponent]
        return {
            "reward": 20,
            "message": f"{agent} shot {opponent}. {opponent} dropped {dropped_items or 'no items'}.",
            "event_type": "shot_opponent",
            "action_success": True,
            "death": True,
            "death_agent": opponent,
            "death_trap_owner": agent,
            "death_trap_type": "gun",
            "death_to_own_trap": False,
            "dropped_items": dropped_items,
        }

    def _available_actions(self, position, agent=None):
        actions = list(self._connected_doors(position).keys())
        for object_name in self.rooms[position]:
            actions.append(f"search_{object_name}")
        for object_name in self.rooms[position]:
            actions.append(f"trap_{object_name}")
        for direction in self._connected_doors(position):
            actions.append(f"trap_door_{direction}")
            actions.append(f"inspect_door_{direction}")
        if position == self.exit_position:
            actions.append("trap_exit_door")
            actions.append("use_exit_door")
        if agent is not None:
            opponent = self.other_agent(agent)
            if (
                "gun" in self.inventories[agent]
                and self.positions[opponent] == position
                and not self._is_dead(opponent)
            ):
                actions.append("shoot")
        return actions

    def _build_doors(self):
        doors = set()
        for y in range(self.size):
            for x in range(self.size):
                room = (x, y)
                for direction in ("right", "down"):
                    neighbor = self._neighbor(room, direction)
                    if self.in_bounds(*neighbor):
                        doors.add(get_door_key(room, neighbor))
        return doors

    def _build_rooms(self):
        rooms = {}
        for y in range(self.size):
            for x in range(self.size):
                start = (x + y * self.size) % len(OBJECT_POOL)
                object_count = 2 + ((x + y) % 2)
                objects = [
                    OBJECT_POOL[(start + offset) % len(OBJECT_POOL)]
                    for offset in range(object_count)
                ]
                rooms[(x, y)] = objects

        # Ensure the fixed item rooms contain the exact objects used above.
        required_objects = {
            (1, 3): "desk",
            (3, 1): "safe",
            (2, 2): "cabinet",
            (0, 4): "safe",
        }
        for position, object_name in required_objects.items():
            if object_name not in rooms[position]:
                rooms[position][-1] = object_name
        return rooms

    def _room_has_trap(self, position):
        for traps in self.object_traps.values():
            for trap_position, _ in traps:
                if trap_position == position:
                    return True
        for traps in self.door_traps.values():
            for trap in traps:
                if trap == self._exit_door_key() and position == self.exit_position:
                    return True
                if isinstance(trap, tuple) and len(trap) == 2 and position in trap:
                    return True
        return False

    def _neighbor(self, position, direction):
        x, y = position
        dx, dy = DIRECTION_DELTAS[direction]
        return (x + dx, y + dy)

    def _connected_doors(self, position):
        connected = {}
        for direction in MOVEMENT_ACTIONS:
            neighbor = self._neighbor(position, direction)
            if self._has_door(position, neighbor):
                connected[direction] = get_door_key(position, neighbor)
        return connected

    def _has_door(self, room_a, room_b):
        return self.in_bounds(*room_b) and get_door_key(room_a, room_b) in self.doors

    def _exit_door_key(self):
        return ("exit_door", self.exit_position)

    def _kill_agent(self, agent, death_position):
        dropped_items = list(self.inventories[agent])
        self.inventories[agent].clear()
        self._drop_items(death_position, dropped_items)
        return dropped_items

    def _drop_items(self, position, dropped_items):
        if not dropped_items:
            return

        target_key = None
        for object_name in self.rooms[position]:
            object_key = (position, object_name)
            if object_key not in self.items:
                target_key = object_key
                break

        if target_key is None:
            target_key = (position, self.rooms[position][0])

        self.items.setdefault(target_key, []).extend(dropped_items)

    def _trap_failure(self, message, event_type, trap_type, door=None):
        return {
            "reward": -3,
            "message": message,
            "event_type": event_type,
            "trap_placed": None,
            "trap_type": trap_type,
            "door": door,
            "action_success": False,
        }

    def _make_public_event(self, info):
        event_type = info["event_type"]
        actor = info["agent"]
        room = info["position_before"]
        base_event = {
            "step": info["current_step"],
            "actor": actor,
            "event_type": event_type,
            "room": room,
            "action": info["action"],
            "object": info.get("searched_object"),
            "found_item": info.get("found_item"),
            "trap_type": info.get("trap_type"),
            "trap_target": info.get("trap_placed"),
            "door": info.get("door"),
        }

        if event_type == "found_item":
            base_event["message"] = (
                f"{actor} searched {info['searched_object']} and found {info['found_item']}."
            )
        elif event_type == "empty_search":
            base_event["message"] = f"{actor} searched {info['searched_object']} and found nothing."
        elif event_type == "trap_placed":
            base_event["message"] = f"{actor} placed a trap on {info['trap_placed']}."
        elif event_type == "door_trap_placed":
            base_event["message"] = f"{actor} placed a trap on {info['trap_placed']} door."
        elif event_type == "exit_door_trap_placed":
            base_event["message"] = f"{actor} placed a trap on exit door."
        elif event_type == "move":
            base_event["message"] = f"{actor} used a door to move."
        elif event_type in ("triggered_trap", "door_trap_triggered", "exit_door_trap_triggered"):
            base_event["message"] = info["message"]
        elif event_type == "shot_opponent":
            base_event["message"] = info["message"]
        elif event_type in ("escaped", "failed_exit_door"):
            base_event["message"] = info["message"]
        else:
            return None

        return base_event
