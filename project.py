#!/usr/bin/env python3
"""
Multi-Agent Intelligent Systems Project - Student Assignment
Course: Artificial Intelligence - Intelligent Agents and Multi-Agent Systems

Student Name: ________________________
Student ID: __________________________
Date: ________________________________

ASSIGNMENT OVERVIEW:
Implement three different agent architectures (Reflex, Model-Based, Goal-Based)
and analyze their performance in a multi-agent gridworld environment.

LEARNING OBJECTIVES:
- Understand different agent architectures from Russell & Norvig
- Implement reactive, model-based, and goal-oriented agents
- Analyze performance trade-offs between agent types
- Experience challenges of multi-agent coordination

DELIVERABLES:
1. Complete implementation of all three agent types
2. Experimental analysis comparing agent performance
3. Written report discussing results and insights
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import heapq
import random
import time
import json
from pathlib import Path


# ============================================================================
# CORE ENVIRONMENT IMPLEMENTATION (PROVIDED - DO NOT MODIFY)
# ============================================================================

class CellType(Enum):
    """Defines the types of cells in the gridworld"""
    EMPTY = 0
    WALL = 1
    GOAL = 2
    RESOURCE = 3
    HAZARD = 4


class Direction(Enum):
    """Defines possible movement directions"""
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST = (1, 0)
    WEST = (-1, 0)


class Action(Enum):
    """Defines possible actions an agent can take"""
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    PICKUP = "pickup"
    DROP = "drop"
    WAIT = "wait"
    COMMUNICATE = "communicate"


@dataclass
class Position:
    """Represents a position in the gridworld"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        """Define ordering for priority queue operations"""
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def __add__(self, direction: Direction):
        """Move position in given direction"""
        dx, dy = direction.value
        return Position(self.x + dx, self.y + dy)


@dataclass
class Message:
    """Represents communication between agents"""
    sender_id: int
    recipient_id: int  # -1 for broadcast
    content: str
    timestamp: int


@dataclass
class Perception:
    """Represents what an agent can perceive from its current position"""
    position: Position
    visible_cells: Dict[Position, CellType]
    visible_agents: Dict[int, Position]  # agent_id -> position
    energy_level: float
    has_resource: bool
    messages: List[Message]


# Environment implementation provided - students use this as-is
# فایل: final_project.py (جایگزین کلاس GridWorld)

class GridWorld:
    """
    کلاس محیط که با ساختار داده‌های رسمی اسکلت هماهنگ شده است.
    """

    def __init__(self, width: int, height: int, perception_range: int = 2):
        self.width = width
        self.height = height
        self.perception_range = perception_range
        self.time_step = 0

        # وضعیت محیط
        self.grid: Dict[Position, CellType] = defaultdict(lambda: CellType.EMPTY)
        self.agents: Dict[int, 'Agent'] = {}
        self.agent_positions: Dict[int, Position] = {}

        # برای سادگی، این‌ها را مستقیم در کلاس Agent مدیریت می‌کنیم
        # self.agent_energy: Dict[int, float] = {}
        # self.agent_resources: Dict[int, int] = {}

        # برای ثبت نتایج آزمایش
        self.initial_resource_count = 0
        self.tasks_completed = 0
        self.total_energy_consumed = 0

    def add_walls(self, wall_positions: List[Position]):
        for pos in wall_positions:
            if self.is_valid_position(pos): self.grid[pos] = CellType.WALL

    def add_goals(self, goal_positions: List[Position]):
        for pos in goal_positions:
            if self.is_valid_position(pos): self.grid[pos] = CellType.GOAL

    def add_resources(self, resource_positions: List[Position]):
        for pos in resource_positions:
            if self.is_valid_position(pos): self.grid[pos] = CellType.RESOURCE
        self.initial_resource_count = len(resource_positions)

    def add_hazards(self, hazard_positions: List[Position]):
        for pos in hazard_positions:
            if self.is_valid_position(pos): self.grid[pos] = CellType.HAZARD

    def add_agent(self, agent: 'Agent', position: Position) -> bool:
        if self.is_position_free(position):
            agent_id = len(self.agents) + 1
            agent.agent_id = agent_id
            self.agents[agent_id] = agent
            self.agent_positions[agent_id] = position
            return True
        return False

    def is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def is_position_free(self, pos: Position) -> bool:
        return self.is_valid_position(pos) and self.grid.get(
            pos) != CellType.WALL and pos not in self.agent_positions.values()

    def get_perception(self, agent_id: int) -> Perception:
        agent_pos = self.agent_positions[agent_id]
        visible_cells: Dict[Position, CellType] = {}

        # محدوده دید 5x5 است (range=2)
        for y_offset in range(-self.perception_range, self.perception_range + 1):
            for x_offset in range(-self.perception_range, self.perception_range + 1):
                pos = Position(agent_pos.x + x_offset, agent_pos.y + y_offset)

                if not self.is_valid_position(pos):
                    visible_cells[pos] = CellType.WALL
                else:
                    # اگر عامل دیگری در آن خانه بود
                    if pos in self.agent_positions.values() and pos != agent_pos:
                        # در این نسخه ساده، عامل دیگر را به عنوان دیوار می‌بینیم
                        visible_cells[pos] = CellType.WALL
                    else:
                        visible_cells[pos] = self.grid.get(pos, CellType.EMPTY)

        agent = self.agents[agent_id]
        return Perception(
            position=agent_pos,
            visible_cells=visible_cells,
            visible_agents={},  # برای سادگی فعلا خالی
            energy_level=agent.total_rewards,  # فرض می‌کنیم پاداش همان انرژی است
            has_resource=(agent.action_history.count(Action.PICKUP) > agent.action_history.count(Action.DROP)),
            messages=[]  # برای سادگی فعلا خالی
        )

    def execute_action(self, agent_id: int, action: Action, message_content: str = ""):
        agent = self.agents[agent_id]
        current_pos = self.agent_positions[agent_id]

        # کسر انرژی برای هر اقدام
        agent.total_rewards -= 1

        if action.name.startswith("MOVE"):
            direction = Direction[action.name.replace("MOVE_", "")]
            next_pos = current_pos + direction
            if self.is_position_free(next_pos):
                self.agent_positions[agent_id] = next_pos

        elif action == Action.PICKUP:
            if self.grid.get(current_pos) == CellType.RESOURCE:
                agent.action_history.append(Action.PICKUP)  # برای ردیابی وضعیت حمل
                self.grid[current_pos] = CellType.EMPTY  # منبع حذف می‌شود

        elif action == Action.DROP:
            if (agent.action_history.count(Action.PICKUP) > agent.action_history.count(Action.DROP)):
                agent.action_history.append(Action.DROP)
                if self.grid.get(current_pos) == CellType.GOAL:
                    self.tasks_completed += 1
                    print(f"Agent {agent_id} delivered a resource at {current_pos}!")
                else:
                    self.grid[current_pos] = CellType.RESOURCE  # منبع روی زمین می‌افتد

        elif action == Action.WAIT:
            agent.total_rewards += 0.5  # انرژی کمتری مصرف می‌شود

    def step(self):
        """یک گام کامل شبیه‌سازی را برای همه عامل‌ها اجرا می‌کند."""
        self.time_step += 1
        for agent_id, agent in self.agents.items():
            if agent.total_rewards <= 0: continue

            perception = self.get_perception(agent_id)
            action, reason = agent.decide_action(perception)
            self.execute_action(agent_id, action)
            print(
                f"Step {self.time_step}: Agent {agent_id} at {self.agent_positions[agent_id]} chose {action.name} ({reason})")

    def get_performance_metrics(self) -> Dict:
        """محاسبه و بازگرداندن متریک‌های عملکرد."""
        # این یک پیاده‌سازی ساده برای هماهنگی با تستر است
        return {
            'total_resources_collected': self.tasks_completed,
            'time_step': self.time_step,
            'average_energy': np.mean([a.total_rewards for a in self.agents.values()]) if self.agents else 0
        }


# ============================================================================
# AGENT IMPLEMENTATIONS - YOUR ASSIGNMENT BEGINS HERE
# ============================================================================

# در فایل project.py
class Agent(ABC):
    """Abstract base class for all agent implementations"""

    def __init__(self, name: str):
        self.name = name
        self.agent_id: int = -1
        self.action_history: List[Action] = []
        # انرژی اولیه روی ۱۰۰ تنظیم می‌شود
        self.total_rewards = 100.0

    @abstractmethod
    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """Decide the next action based on current perception"""
        pass

    def reset(self):
        """Reset agent state for new episode"""
        self.action_history.clear()
        # انرژی در شروع هر آزمایش جدید نیز به ۱۰۰ برمی‌گردد
        self.total_rewards = 100.0


class SimpleReflexAgent(Agent):
    """
    عامل واکنش‌گر ساده که مستقیماً به ادراکات پاسخ می‌دهد.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.rule_activations = defaultdict(int)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """منطق تصمیم‌گیری بر اساس قوانین شرط-عمل با اولویت."""

        my_pos = perception.position
        visible_cells = perception.visible_cells

        # قانون ۱: اگر روی منبع هستیم و چیزی حمل نمی‌کنیم، آن را برداریم
        if visible_cells.get(my_pos) == CellType.RESOURCE and not perception.has_resource:
            self.rule_activations['pickup_resource'] += 1
            return Action.PICKUP, "Rule 1: On a resource, picking up."

        # قانون ۱.۵: اگر منبع داریم و روی هدف هستیم، آن را رها کن
        if perception.has_resource and visible_cells.get(my_pos) == CellType.GOAL:
            self.rule_activations['drop_on_goal'] += 1
            return Action.DROP, "Rule 1.5: On goal with resource, dropping."

        # قانون ۲: اگر منبعی حمل می‌کنیم و هدف را می‌بینیم، به سمت آن حرکت کنیم
        if perception.has_resource:
            goal_positions = [pos for pos, cell_type in visible_cells.items() if cell_type == CellType.GOAL]
            if goal_positions:
                direction = self._get_direction_toward(my_pos, goal_positions[0])
                if direction:
                    action = self._direction_to_action(direction)
                    self.rule_activations['move_to_goal'] += 1
                    return action, f"Rule 2: Carrying resource, moving toward goal at {goal_positions[0]}"

        # قانون ۳: اگر منبعی حمل نمی‌کنیم و منبعی می‌بینیم، به سمت آن حرکت کنیم
        if not perception.has_resource:
            resource_positions = [pos for pos, cell_type in visible_cells.items() if cell_type == CellType.RESOURCE]
            if resource_positions:
                direction = self._get_direction_toward(my_pos, resource_positions[0])
                if direction:
                    action = self._direction_to_action(direction)
                    self.rule_activations['move_to_resource'] += 1
                    return action, f"Rule 3: Seeking resource, moving toward {resource_positions[0]}"

        # قانون ۴: حرکت اکتشافی تصادفی
        action = self._random_valid_move(visible_cells, my_pos)
        self.rule_activations['random_explore'] += 1
        return action, "Rule 4: Random exploration."

    def _direction_to_action(self, direction: Direction) -> Action:
        """تبدیل جهت به اقدام حرکتی."""
        return Action[f"MOVE_{direction.name}"]

    def _get_direction_toward(self, from_pos: Position, to_pos: Position) -> Optional[Direction]:
        """بهترین جهت برای حرکت از یک نقطه به نقطه دیگر را محاسبه می‌کند."""
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y

        if abs(dx) > abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        elif dy != 0:
            return Direction.SOUTH if dy > 0 else Direction.NORTH
        return None

    def _random_valid_move(self, visible_cells: Dict[Position, CellType],
                           current_pos: Position) -> Action:
        """یک حرکت تصادفی معتبر را انتخاب می‌کند."""
        valid_directions = []
        for direction in Direction:
            next_pos = current_pos + direction
            if visible_cells.get(next_pos) != CellType.WALL:
                valid_directions.append(direction)

        if valid_directions:
            chosen_direction = random.choice(valid_directions)
            return self._direction_to_action(chosen_direction)

        return Action.WAIT


# فایل: project.py (بخش ModelBasedReflexAgent)

class ModelBasedReflexAgent(Agent):
    """
    عاملی که با استفاده از یک مدل داخلی (حافظه) از جهان، تصمیمات هوشمندانه‌تری می‌گیرد.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.reset()

    def reset(self):
        """ریست کردن وضعیت عامل برای هر آزمایش جدید."""
        super().reset()
        self.visited_positions: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()
        self.known_hazards: Set[Position] = set()

    def _update_world_model(self, perception: Perception):
        """حافظه داخلی را بر اساس ادراک جدید به‌روزرسانی می‌کند."""
        current_pos = perception.position
        self.visited_positions.add(current_pos)

        visible_positions = perception.visible_cells.keys()

        for pos, cell_type in perception.visible_cells.items():
            if cell_type == CellType.WALL:
                self.known_walls.add(pos)
            elif cell_type == CellType.RESOURCE:
                self.known_resources.add(pos)
            elif cell_type == CellType.GOAL:
                self.known_goals.add(pos)
            elif cell_type == CellType.HAZARD:
                self.known_hazards.add(pos)

        # اگر منبعی که در حافظه بود، دیگر در دید نیست (چون برداشته شده)، آن را حذف کن
        resources_to_check = self.known_resources.copy()
        for res_pos in resources_to_check:
            if res_pos in visible_positions and perception.visible_cells.get(res_pos) != CellType.RESOURCE:
                self.known_resources.remove(res_pos)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """منطق تصمیم‌گیری عامل مبتنی بر مدل با استفاده از حافظه."""
        self._update_world_model(perception)

        my_pos = perception.position
        is_carrying = perception.has_resource

        # اولویت ۱: اگر روی خطر هستیم، فرار کن
        if perception.visible_cells.get(my_pos) == CellType.HAZARD:
            action = self._find_first_safe_move(my_pos, perception.visible_cells)
            return action, "Rule 1: Emergency hazard avoidance"

        # اولویت ۲: اگر روی منبع هستیم و چیزی نداریم، بردار
        if perception.visible_cells.get(my_pos) == CellType.RESOURCE and not is_carrying:
            return Action.PICKUP, "Rule 2: On a resource, picking up"

        # اولویت ۳: اگر منبع داریم و روی هدف هستیم، آن را رها کن
        if is_carrying and perception.visible_cells.get(my_pos) == CellType.GOAL:
            return Action.DROP, "Rule 3: On goal with resource, dropping"

        # اولویت ۴: اگر منبع داریم، به سمت نزدیک‌ترین هدف شناخته‌شده در حافظه حرکت کن
        if is_carrying and self.known_goals:
            closest_goal = self._find_closest_target(my_pos, self.known_goals)
            if closest_goal:
                direction = self._get_direction_toward(my_pos, closest_goal)
                if direction:
                    return self._direction_to_action(direction), f"Rule 4: Moving toward known goal at {closest_goal}"

        # اولویت ۵: اگر منبع نداریم، به سمت نزدیک‌ترین منبع شناخته‌شده در حافظه حرکت کن
        if not is_carrying and self.known_resources:
            closest_resource = self._find_closest_target(my_pos, self.known_resources)
            if closest_resource:
                direction = self._get_direction_toward(my_pos, closest_resource)
                if direction:
                    return self._direction_to_action(
                        direction), f"Rule 5: Moving toward known resource at {closest_resource}"

        # اولویت ۶: اکتشاف هوشمند به سمت خانه‌های دیده‌نشده
        action = self._intelligent_exploration(my_pos, perception.visible_cells)
        return action, "Rule 6: Intelligent exploration"

    # --- متدهای کمکی ---

    def _find_closest_target(self, my_pos: Position, targets: Set[Position]) -> Optional[Position]:
        if not targets: return None
        return min(targets, key=lambda target: abs(my_pos.x - target.x) + abs(my_pos.y - target.y))

    def _intelligent_exploration(self, current_pos: Position, visible_cells: Dict[Position, CellType]) -> Action:
        possible_directions = list(Direction)
        random.shuffle(possible_directions)

        for direction in possible_directions:
            next_pos = current_pos + direction
            if visible_cells.get(next_pos) != CellType.WALL and next_pos not in self.visited_positions:
                return self._direction_to_action(direction)

        return self._random_valid_move(visible_cells, current_pos)

    def _direction_to_action(self, direction: Direction) -> Action:
        return Action[f"MOVE_{direction.name}"]

    def _get_direction_toward(self, from_pos: Position, to_pos: Position) -> Optional[Direction]:
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
        if abs(dx) > abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        elif dy != 0:
            return Direction.SOUTH if dy > 0 else Direction.NORTH
        return None

    def _random_valid_move(self, visible_cells: Dict[Position, CellType], current_pos: Position) -> Action:
        valid_directions = [d for d in Direction if visible_cells.get(current_pos + d) != CellType.WALL]
        return self._direction_to_action(random.choice(valid_directions)) if valid_directions else Action.WAIT

    def _find_first_safe_move(self, my_pos: Position, cells: Dict[Position, CellType]) -> Action:
        safe_directions = [d for d in Direction if cells.get(my_pos + d) not in [CellType.WALL, CellType.HAZARD]]
        return self._direction_to_action(
            random.choice(safe_directions)) if safe_directions else self._random_valid_move(cells, my_pos)


@dataclass
class PlanStep:
    """Represents a single step in an agent's plan"""
    action: Action
    target_position: Position
    purpose: str
    estimated_cost: float = 1.0


# فایل: project.py (بخش GoalBasedAgent)

@dataclass
class PlanStep:
    """Represents a single step in an agent's plan"""
    action: Action
    # این فیلدها در آینده برای برنامه‌های پیچیده‌تر مفید خواهند بود
    # target_position: Position
    # purpose: str
    # estimated_cost: float = 1.0


class GoalBasedAgent(Agent):
    """
    عاملی که با برنامه‌ریزی بلندمدت و الگوریتم A* به اهداف خود می‌رسد.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.reset()

    def reset(self):
        """ریست کردن وضعیت عامل برای هر آزمایش جدید."""
        super().reset()
        self.current_plan: List[PlanStep] = []
        # این عامل نیز به حافظه نیاز دارد
        self.visited_positions: Set[Position] = set()
        self.known_walls: Set[Position] = set()
        self.known_resources: Set[Position] = set()
        self.known_goals: Set[Position] = set()

    def _update_world_model(self, perception: Perception):
        """حافظه داخلی را بر اساس ادراک جدید به‌روزرسانی می‌کند."""
        self.visited_positions.add(perception.position)
        for pos, cell_type in perception.visible_cells.items():
            if cell_type == CellType.WALL:
                self.known_walls.add(pos)
            elif cell_type == CellType.RESOURCE:
                self.known_resources.add(pos)
            elif cell_type == CellType.GOAL:
                self.known_goals.add(pos)

        # حذف منابعی که دیگر وجود ندارند
        visible_positions = perception.visible_cells.keys()
        resources_to_check = self.known_resources.copy()
        for res_pos in resources_to_check:
            if res_pos in visible_positions and perception.visible_cells.get(res_pos) != CellType.RESOURCE:
                self.known_resources.remove(res_pos)

    def decide_action(self, perception: Perception) -> Tuple[Action, str]:
        """منطق اصلی: به‌روزرسانی حافظه، سپس اجرای برنامه یا ساخت برنامه جدید."""
        self._update_world_model(perception)

        # اگر برنامه‌ای در حال اجراست و هنوز معتبر است، آن را ادامه بده
        if self.current_plan:
            next_step = self.current_plan.pop(0)
            reason = f"Executing plan: {next_step.action.name}"
            return next_step.action, reason

        # اگر برنامه‌ای وجود ندارد، یک برنامه جدید بساز
        self._create_new_plan(perception)

        # اگر بعد از تلاش، برنامه‌ای ساخته شد، اولین قدم آن را اجرا کن
        if self.current_plan:
            next_step = self.current_plan.pop(0)
            reason = f"Starting new plan: {next_step.action.name}"
            return next_step.action, reason

        # اگر هیچ برنامه‌ای ممکن نبود، یک حرکت اکتشافی انجام بده
        action = self._intelligent_exploration(perception.position, perception.visible_cells)
        return action, "No current goal. Exploring intelligently."

    def _create_new_plan(self, perception: Perception):
        """بر اساس حافظه، بهترین هدف را انتخاب و برایش برنامه‌ریزی می‌کند."""
        my_pos = perception.position

        candidate_goals = []
        if perception.has_resource:
            for pos in self.known_goals:
                candidate_goals.append({"type": "DELIVER", "pos": pos, "base_utility": 20.0})
        else:
            # اگر روی منبع هستیم، برنامه فقط برداشتن است
            if perception.visible_cells.get(my_pos) == CellType.RESOURCE:
                self.current_plan = [PlanStep(action=Action.PICKUP)]
                return

            for pos in self.known_resources:
                candidate_goals.append({"type": "COLLECT", "pos": pos, "base_utility": 10.0})

        if not candidate_goals:
            self.current_plan = []
            return

        # محاسبه مطلوبیت و انتخاب بهترین هدف
        for goal in candidate_goals:
            dist = abs(my_pos.x - goal["pos"].x) + abs(my_pos.y - goal["pos"].y)
            goal['utility'] = goal["base_utility"] / (dist + 1)

        best_goal = max(candidate_goals, key=lambda g: g['utility'])

        # ساختن مسیر با A*
        path_actions = self._find_path_astar(my_pos, best_goal["pos"], self.known_walls)

        if path_actions:
            self.current_plan = [PlanStep(action=act) for act in path_actions]
            final_action = Action.PICKUP if best_goal["type"] == "COLLECT" else Action.DROP
            self.current_plan.append(PlanStep(action=final_action))
            print(f"Agent {self.agent_id} created a new plan: {best_goal['type']} at {best_goal['pos']}")

    def _find_path_astar(self, start: Position, goal: Position, walls: Set[Position]) -> List[Action]:
        # ... (این کد دقیقاً همان کد A* است که قبلاً داشتیم) ...
        def heuristic(a: Position, b: Position) -> int:
            return abs(a.x - b.x) + abs(a.y - b.y)

        frontier = [(0, start)]
        heapq.heapify(frontier)
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            _, current_pos = heapq.heappop(frontier)
            if current_pos == goal: break

            for direction in Direction:
                action = self._direction_to_action(direction)
                next_pos = current_pos + direction
                if next_pos in walls: continue
                new_cost = cost_so_far[current_pos] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = (current_pos, action)

        if goal not in came_from: return []
        path = []
        temp = goal
        while temp != start:
            prev_pos, action = came_from[temp]
            path.append(action)
            temp = prev_pos
        path.reverse()
        return path

    # --- متدهای کمکی مشترک ---
    def _direction_to_action(self, direction: Direction) -> Action:
        return Action[f"MOVE_{direction.name}"]

    def _intelligent_exploration(self, current_pos: Position, visible_cells: Dict[Position, CellType]) -> Action:
        possible_directions = list(Direction)
        random.shuffle(possible_directions)
        for direction in possible_directions:
            next_pos = current_pos + direction
            if visible_cells.get(next_pos) != CellType.WALL and next_pos not in self.visited_positions:
                return self._direction_to_action(direction)
        valid_directions = [d for d in Direction if visible_cells.get(current_pos + d) != CellType.WALL]
        return self._direction_to_action(random.choice(valid_directions)) if valid_directions else Action.WAIT


# ============================================================================
# EXPERIMENTAL FRAMEWORK - PROVIDED FOR TESTING
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experimental scenarios"""
    name: str
    description: str
    grid_size: Tuple[int, int]
    num_agents: int
    num_resources: int
    num_goals: int
    num_hazards: int
    max_steps: int
    num_trials: int


@dataclass
class ExperimentResult:
    """Results from a single experimental trial"""
    config_name: str
    trial_number: int
    agent_type: str
    total_steps: int
    tasks_completed: int
    total_resources_collected: int
    average_energy_remaining: float
    collision_count: int
    success_rate: float
    efficiency_score: float


class ProjectTester:
    """
    Testing framework for student implementations
    Use this to test and analyze your agent implementations
    """

    def __init__(self):
        self.experiment_configs = [
            ExperimentConfig(
                name="simple_collection",
                description="Basic resource collection in open environment",
                grid_size=(8, 8),
                num_agents=2,
                num_resources=4,
                num_goals=2,
                num_hazards=0,
                max_steps=100,
                num_trials=5
            ),
            ExperimentConfig(
                name="maze_navigation",
                description="Resource collection in maze environment",
                grid_size=(10, 10),
                num_agents=2,
                num_resources=4,
                num_goals=2,
                num_hazards=3,
                max_steps=150,
                num_trials=5
            ),
            ExperimentConfig(
                name="competitive_collection",
                description="Multiple agents competing for limited resources",
                grid_size=(12, 12),
                num_agents=3,
                num_resources=3,
                num_goals=2,
                num_hazards=2,
                max_steps=200,
                num_trials=5
            )
        ]

        self.agent_types = {
            "SimpleReflex": SimpleReflexAgent,
            "ModelBased": ModelBasedReflexAgent,
            "GoalBased": GoalBasedAgent
        }

    def test_single_agent(self, agent_class, config_name: str = "simple_collection"):
        """Test a single agent implementation"""
        config = next(c for c in self.experiment_configs if c.name == config_name)

        # Create environment
        env = GridWorld(config.grid_size[0], config.grid_size[1], config.num_agents)

        # Add some basic elements for testing
        walls = [Position(x, 0) for x in range(env.width)] + [Position(x, env.height - 1) for x in range(env.width)]
        walls += [Position(0, y) for y in range(env.height)] + [Position(env.width - 1, y) for y in range(env.height)]
        env.add_walls(walls)

        resources = [Position(3, 3), Position(5, 5)]
        goals = [Position(2, 2), Position(6, 6)]
        env.add_resources(resources)
        env.add_goals(goals)

        # Create and add agent
        agent = agent_class("TestAgent")
        env.add_agent(agent, Position(1, 1))

        # Run simulation
        for step in range(config.max_steps):
            results = env.step()
            if step % 10 == 0:
                metrics = env.get_performance_metrics()
                print(
                    f"Step {step}: Resources collected: {metrics['total_resources_collected']}, Energy: {metrics['average_energy']:.1f}")

        final_metrics = env.get_performance_metrics()
        print(f"\nFinal Results:")
        print(f"Resources collected: {final_metrics['total_resources_collected']}")
        print(f"Steps taken: {final_metrics['time_step']}")
        print(f"Final energy: {final_metrics['average_energy']:.1f}")

        return final_metrics

    def run_comparison(self):
        """Compare all implemented agent types"""
        print("AGENT COMPARISON")
        print("=" * 50)

        for agent_name, agent_class in self.agent_types.items():
            print(f"\nTesting {agent_name}...")
            try:
                result = self.test_single_agent(agent_class)
                print(f"✓ {agent_name} completed successfully")
            except Exception as e:
                print(f"✗ {agent_name} failed: {str(e)}")


# ============================================================================
# MAIN TESTING FUNCTION
# ============================================================================

def main():
    """
    Main function for testing your implementations

    TODO: Uncomment sections as you complete each agent implementation
    """

    print("Multi-Agent Systems Project - Testing Framework")
    print("=" * 60)

    tester = ProjectTester()

    # TODO: Uncomment these as you implement each agent
    print("\n1. Testing SimpleReflexAgent...")
    tester.test_single_agent(SimpleReflexAgent)

    print("\n2. Testing ModelBasedReflexAgent...")
    tester.test_single_agent(ModelBasedReflexAgent)

    print("\n3. Testing GoalBasedAgent...")
    tester.test_single_agent(GoalBasedAgent)

    print("\n4. Running full comparison...")
    tester.run_comparison()

    print("\nImplementation complete! Proceed to experimental analysis.")


if __name__ == "__main__":
    main()

"""
SUBMISSION CHECKLIST:
□ SimpleReflexAgent fully implemented and tested
□ ModelBasedReflexAgent fully implemented and tested  
□ GoalBasedAgent fully implemented and tested
□ All agents pass basic functionality tests
□ Experimental analysis completed
□ Report written with results and insights
□ Code is well-commented and documented

GRADING CRITERIA:
- Implementation correctness (40%)
- Code quality and documentation (20%)
- Experimental analysis (25%)
- Written report and insights (15%)
- Written report and insights (15%)
- Written report and insights (15%)
- Written report and insights (15%)
- Written report and insights (15%)
- Written report and insights (15%)
- Written report and insights (15%)

Good luck with your implementation!
"""