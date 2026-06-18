"""Simple agents for the Agent vs Agent text environment."""

import json
import random

from config import get_llm_config
from env import DIRECTION_DELTAS, MOVEMENT_ACTIONS, get_door_key


def position_key(position):
    return str(tuple(position))


class BaseAgent:
    def __init__(self, name):
        self.name = name
        self.memory = self._new_memory()

    def _new_memory(self):
        return {
            "visited_rooms": [],
            "searched_objects": {},
            "known_empty_objects": [],
            "known_item_locations": {},
            "collected_items": [],
            "suspected_enemy_traps": [],
            "own_traps": [],
            "doors_used_safely": [],
            "suspected_trapped_doors": [],
            "own_door_traps": [],
            "own_exit_door_traps": [],
            "door_trap_triggers": [],
            "exit_door_suspected_dangerous": False,
            "chokepoint_doors": [],
            "observed_opponent_events": [],
            "observed_opponent_items": {},
            "observed_opponent_traps": [],
            "opponent_searched_objects": {},
            "opponent_failed_searches": [],
            "opponent_last_seen_action": None,
            "last_seen_opponent": None,
            "action_history": [],
            "failed_actions": [],
            "important_events": [],
        }

    def act(self, observation):
        raise NotImplementedError

    def observe_result(self, observation, action, reward, info):
        position = tuple(observation["position"])
        room_key = position_key(position)

        self._process_observed_events(observation.get("recent_observed_events", []))
        self._remember_unique("visited_rooms", position)

        if observation["opponent_visible"]:
            self.memory["last_seen_opponent"] = {
                "position": tuple(observation["opponent_position"]),
                "step": observation["current_step"],
            }

        for object_name in observation.get("known_enemy_traps", []):
            self._remember_unique("suspected_enemy_traps", (position, object_name))

        searched_object = info.get("searched_object")
        if searched_object:
            searched = self.memory["searched_objects"].setdefault(room_key, [])
            if searched_object not in searched:
                searched.append(searched_object)

        event_type = info.get("event_type")
        if event_type == "empty_search" and searched_object:
            self._remember_unique("known_empty_objects", (position, searched_object))

        found_items = info.get("found_items") or ([info["found_item"]] if info.get("found_item") else [])
        if found_items and searched_object:
            for found_item in found_items:
                self.memory["known_item_locations"][found_item] = (position, searched_object)
                self._remember_unique("collected_items", found_item)

        trap_placed = info.get("trap_placed")
        trap_type = info.get("trap_type")
        door = info.get("door")

        if trap_placed and trap_type == "object":
            self._remember_unique("own_traps", (position, trap_placed))
        elif trap_placed and trap_type == "door":
            self._remember_unique("own_door_traps", door)
        elif trap_placed and trap_type == "exit_door":
            self._remember_unique("own_exit_door_traps", door)

        if info.get("movement_success") and door:
            self._remember_unique("doors_used_safely", door)

        if info.get("door_status") == "known safe" and door:
            self._remember_unique("doors_used_safely", door)
        elif info.get("door_status") == "suspected trapped" and door:
            self._remember_unique("suspected_trapped_doors", door)

        if info.get("triggered_trap") and searched_object:
            self._remember_unique("suspected_enemy_traps", (position, searched_object))
            self._remember_unique(
                "important_events",
                {
                    "step": info["current_step"],
                    "event": "triggered_trap",
                    "position": position,
                    "object": searched_object,
                },
            )
        elif info.get("event_type") == "door_trap_triggered":
            self._remember_unique("suspected_trapped_doors", door)
            self._remember_unique("door_trap_triggers", {
                "step": info["current_step"],
                "door": door,
                "trap_owner": info.get("trap_owner"),
            })
            self._remember_unique("important_events", {
                "step": info["current_step"],
                "event": "door_trap_triggered",
                "door": door,
            })
        elif info.get("event_type") == "exit_door_trap_triggered":
            self.memory["exit_door_suspected_dangerous"] = True
            self._remember_unique("door_trap_triggers", {
                "step": info["current_step"],
                "door": door,
                "trap_owner": info.get("trap_owner"),
            })
            self._remember_unique("important_events", {
                "step": info["current_step"],
                "event": "exit_door_trap_triggered",
            })

        if not info.get("action_success") and not info.get("timeout_skip"):
            self._remember_unique("failed_actions", self._failure_label(action, info))

        self.memory["action_history"].append({
            "step": info["current_step"],
            "position": position,
            "action": action,
            "reward": reward,
            "result": event_type,
            "message": info.get("message"),
        })

    def memory_summary(self):
        return {
            "visited_rooms": self.memory["visited_rooms"],
            "searched_objects": self.memory["searched_objects"],
            "known_empty_objects": self.memory["known_empty_objects"],
            "known_item_locations": self.memory["known_item_locations"],
            "collected_items": self.memory["collected_items"],
            "suspected_enemy_traps": self.memory["suspected_enemy_traps"],
            "own_traps": self.memory["own_traps"],
            "doors_used_safely": self.memory["doors_used_safely"],
            "suspected_trapped_doors": self.memory["suspected_trapped_doors"],
            "own_door_traps": self.memory["own_door_traps"],
            "own_exit_door_traps": self.memory["own_exit_door_traps"],
            "door_trap_triggers": self.memory["door_trap_triggers"],
            "exit_door_suspected_dangerous": self.memory["exit_door_suspected_dangerous"],
            "chokepoint_doors": self.memory["chokepoint_doors"],
            "observed_opponent_events": self.memory["observed_opponent_events"][-8:],
            "observed_opponent_items": self.memory["observed_opponent_items"],
            "observed_opponent_traps": self.memory["observed_opponent_traps"],
            "opponent_searched_objects": self.memory["opponent_searched_objects"],
            "opponent_failed_searches": self.memory["opponent_failed_searches"],
            "opponent_last_seen_action": self.memory["opponent_last_seen_action"],
            "last_seen_opponent": self.memory["last_seen_opponent"],
            "failed_actions": self.memory["failed_actions"][-5:],
            "important_events": self.memory["important_events"][-5:],
        }

    def recent_action_history(self, limit=8):
        return self.memory["action_history"][-limit:]

    def _searched_in_room(self, position):
        return self.memory["searched_objects"].get(position_key(position), [])

    def _suspected_traps_in_room(self, position):
        return [
            object_name
            for trap_position, object_name in self.memory["suspected_enemy_traps"]
            if tuple(trap_position) == tuple(position)
        ]

    def _opponent_searched_in_room(self, position):
        return self.memory["opponent_searched_objects"].get(position_key(position), [])

    def _opponent_failed_searches_in_room(self, position):
        return [
            object_name
            for failed_position, object_name in self.memory["opponent_failed_searches"]
            if tuple(failed_position) == tuple(position)
        ]

    def _door_key(self, position, direction):
        x, y = position
        dx, dy = DIRECTION_DELTAS[direction]
        return get_door_key(position, (x + dx, y + dy))

    def _remember_unique(self, key, value):
        if value not in self.memory[key]:
            self.memory[key].append(value)

    def _process_observed_events(self, events):
        for event in events:
            room = tuple(event["room"])
            actor = event["actor"]
            object_name = event.get("object")
            event_type = event["event_type"]

            self._remember_unique("observed_opponent_events", event)
            self.memory["opponent_last_seen_action"] = {
                "step": event["step"],
                "action": event["action"],
                "room": room,
            }

            if object_name:
                searched = self.memory["opponent_searched_objects"].setdefault(position_key(room), [])
                if object_name not in searched:
                    searched.append(object_name)

            if event_type == "found_item" and event.get("found_item"):
                items = self.memory["observed_opponent_items"].setdefault(actor, [])
                if event["found_item"] not in items:
                    items.append(event["found_item"])
            elif event_type == "empty_search" and object_name:
                self._remember_unique("opponent_failed_searches", (room, object_name))
            elif event_type == "trap_placed":
                target = event.get("trap_target")
                self._remember_unique("observed_opponent_traps", {
                    "room": room,
                    "target": target,
                    "owner": actor,
                    "trap_type": "object",
                })
                if target:
                    self._remember_unique("suspected_enemy_traps", (room, target))
            elif event_type == "door_trap_placed":
                self._remember_unique("observed_opponent_traps", {
                    "room": room,
                    "target": event.get("door"),
                    "owner": actor,
                    "trap_type": "door",
                })
                if event.get("door"):
                    self._remember_unique("suspected_trapped_doors", event["door"])
            elif event_type == "exit_door_trap_placed":
                self.memory["exit_door_suspected_dangerous"] = True
                self._remember_unique("observed_opponent_traps", {
                    "room": room,
                    "target": "exit_door",
                    "owner": actor,
                    "trap_type": "exit_door",
                })

    def _failure_label(self, action, info):
        if info.get("event_type") == "failed_escape":
            return "old escape action failed"
        if info.get("event_type") == "failed_exit_door":
            return "use_exit_door without all items or away from exit room"
        return f"{action}: {info.get('message')}"


class RandomAgent(BaseAgent):
    def act(self, observation):
        if observation.get("is_dead"):
            return "wait"
        return random.choice(observation["available_actions"])


class RuleAgent(BaseAgent):
    def act(self, observation):
        if observation.get("is_dead"):
            return "wait"

        position = tuple(observation["position"])

        if "shoot" in observation["available_actions"]:
            return "shoot"

        if observation["has_all_items"] and observation["at_exit"]:
            if self.memory["exit_door_suspected_dangerous"]:
                return self._avoid_exit_door(observation)
            return "use_exit_door"

        if observation["has_all_items"]:
            return self._move_toward(position, observation["exit_position"], observation)

        unsearched_objects = self._safe_unsearched_objects(observation)

        trap_action = self._strategic_trap_action(observation)
        if trap_action:
            return trap_action

        if unsearched_objects:
            return f"search_{random.choice(unsearched_objects)}"

        return self._move_to_promising_room(observation)

    def _safe_unsearched_objects(self, observation):
        position = tuple(observation["position"])
        searched_objects = self._searched_in_room(position)
        suspected_traps = self._suspected_traps_in_room(position)
        opponent_empty_objects = self._opponent_failed_searches_in_room(position)
        visible_traps = observation.get("known_enemy_traps", [])
        return [
            object_name
            for object_name in observation["objects"]
            if object_name not in searched_objects
            and object_name not in opponent_empty_objects
            and object_name not in suspected_traps
            and object_name not in visible_traps
        ]

    def _strategic_trap_action(self, observation):
        if observation["traps_left"] <= 0:
            return None

        position = tuple(observation["position"])
        if observation["has_exit_door"] and self._opponent_recently_nearby(observation):
            if "trap_exit_door" in observation["available_actions"] and self._exit_door_key(observation) not in self.memory["own_exit_door_traps"]:
                return "trap_exit_door"

        if not self._opponent_recently_nearby(observation):
            return None

        preferred_direction = self._direction_toward_opponent(observation)
        directions = list(observation["connected_doors"])
        if preferred_direction in directions:
            directions.remove(preferred_direction)
            directions.insert(0, preferred_direction)

        for direction in directions:
            action = f"trap_door_{direction}"
            door_key = self._door_key(position, direction)
            if action in observation["available_actions"] and door_key not in self.memory["own_door_traps"]:
                self._remember_unique("chokepoint_doors", door_key)
                return action

        unsearched_objects = self._safe_unsearched_objects(observation)
        if unsearched_objects:
            return f"trap_{random.choice(unsearched_objects)}"
        return None

    def _avoid_exit_door(self, observation):
        position = tuple(observation["position"])
        valid_moves = [
            action
            for action in MOVEMENT_ACTIONS
            if observation["nearby"][action] != "no door"
            and self._door_key(position, action) not in self.memory["suspected_trapped_doors"]
        ]
        if valid_moves:
            return random.choice(valid_moves)

        non_exit_actions = [
            action
            for action in observation["available_actions"]
            if action != "use_exit_door"
        ]
        return random.choice(non_exit_actions) if non_exit_actions else "use_exit_door"

    def _opponent_recently_nearby(self, observation):
        if observation["opponent_visible"]:
            return True

        last_seen = self.memory["last_seen_opponent"]
        if not last_seen:
            return False

        age = observation["current_step"] - last_seen["step"]
        position = tuple(observation["position"])
        opponent_position = tuple(last_seen["position"])
        distance = abs(position[0] - opponent_position[0]) + abs(position[1] - opponent_position[1])
        return age <= 4 and distance <= 2

    def _direction_toward_opponent(self, observation):
        if not observation["opponent_visible"]:
            return None

        x, y = observation["position"]
        opponent_x, opponent_y = observation["opponent_position"]
        if opponent_x > x:
            return "right"
        if opponent_x < x:
            return "left"
        if opponent_y > y:
            return "down"
        if opponent_y < y:
            return "up"
        return None

    def _move_to_promising_room(self, observation):
        position = tuple(observation["position"])
        best_action = None
        best_score = None
        recent_positions = [
            tuple(entry["position"])
            for entry in self.memory["action_history"][-8:]
            if entry["result"] == "move"
        ]
        previous_position = recent_positions[-1] if recent_positions else None

        for action in MOVEMENT_ACTIONS:
            if observation["nearby"][action] == "no door":
                continue
            if self._door_key(position, action) in self.memory["suspected_trapped_doors"]:
                continue

            next_position = self._next_position(position, action)
            searched_count = len(self._searched_in_room(next_position))
            visit_count = self.memory["visited_rooms"].count(next_position)
            recent_visit_penalty = recent_positions.count(next_position) * 3
            backtrack_penalty = 5 if previous_position == next_position else 0
            score = searched_count + visit_count + recent_visit_penalty + backtrack_penalty

            if best_score is None or score < best_score:
                best_action = action
                best_score = score

        if best_action:
            return best_action

        valid_moves = [
            action
            for action in MOVEMENT_ACTIONS
            if observation["nearby"][action] != "no door"
        ]
        return random.choice(valid_moves) if valid_moves else random.choice(observation["available_actions"])

    def _move_toward(self, position, target, observation):
        x, y = position
        target_x, target_y = target
        preferred_actions = []

        if target_x > x:
            preferred_actions.append("right")
        elif target_x < x:
            preferred_actions.append("left")

        if target_y > y:
            preferred_actions.append("down")
        elif target_y < y:
            preferred_actions.append("up")

        for action in preferred_actions:
            if observation["nearby"][action] != "no door" and self._door_key(position, action) not in self.memory["suspected_trapped_doors"]:
                return action

        valid_moves = [
            action
            for action in MOVEMENT_ACTIONS
            if observation["nearby"][action] != "no door"
            and self._door_key(position, action) not in self.memory["suspected_trapped_doors"]
        ]
        return random.choice(valid_moves) if valid_moves else random.choice(observation["available_actions"])

    def _next_position(self, position, action):
        x, y = position
        if action == "up":
            return (x, y - 1)
        if action == "down":
            return (x, y + 1)
        if action == "left":
            return (x - 1, y)
        return (x + 1, y)

    def _exit_door_key(self, observation):
        return ("exit_door", tuple(observation["exit_room"]))


class LLMAgent(BaseAgent):
    def __init__(self, name, client_func=None, llm_config=None):
        super().__init__(name)
        self.client_func = client_func
        self.llm_config = llm_config or get_llm_config()

    def act(self, observation):
        if observation.get("is_dead"):
            self._last_llm_reason = "Death timeout."
            return "wait"

        prompt = self._build_prompt(observation)
        action = None
        reason = "No LLM client configured."

        if self.client_func is not None:
            try:
                response_text = self.client_func(prompt)
                parsed = json.loads(response_text)
                action = parsed.get("action")
                reason = parsed.get("reason", "")
            except (RuntimeError, TypeError, json.JSONDecodeError, AttributeError):
                action = None
                reason = "LLM response could not be parsed."

        if action not in observation["available_actions"]:
            action = random.choice(observation["available_actions"])
            reason = "Fallback random action."

        self._last_llm_reason = reason
        return action

    def _build_prompt(self, observation):
        prompt_data = {
            "current_observation": observation,
            "memory_summary": self.memory_summary(),
            "recent_action_history": self.recent_action_history(),
            "available_actions": observation["available_actions"],
            "remaining_turns": observation["turns_left"],
            "model": self.llm_config["model"],
        }
        return (
            "You are playing a text/grid Agent vs Agent game.\n"
            "Each grid tile is a room. Items and traps are hidden inside room objects.\n"
            "Rooms are connected by doors. Moving up/down/left/right means using a door.\n"
            "Doors can be trapped, and using an opponent-trapped door sends you back to start.\n"
            "The exit is a special exit door in the exit room. Use use_exit_door to win.\n"
            "The gun is a required item. If you have the gun and share a room with the opponent, shoot can kill them.\n"
            "The exit door can also be trapped; trapping it can be useful because opponents must use it to win.\n"
            "You may observe the opponent's actions only when both spies are in the same room.\n"
            "Use observed events to infer opponent inventory, dangerous traps, and searched objects.\n"
            "If you saw the opponent place a trap, avoid that target.\n"
            "If you saw the opponent find an item, assume that item is gone.\n"
            "Use your memory to avoid repeated searches, remember empty objects, "
            "track suspected enemy traps, and plan efficient movement through doors.\n"
            "Choose only one action from available_actions.\n"
            "Return only JSON in this exact shape:\n"
            '{"action": "search_desk", "reason": "The desk in this room has not been searched yet."}\n\n'
            f"{json.dumps(prompt_data, indent=2)}"
        )
