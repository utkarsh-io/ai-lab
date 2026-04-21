

from __future__ import annotations

import argparse
import itertools
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Iterable, Optional, Any


@dataclass
class Assignment:
    """Represents a single assignment node in the scheduling graph."""

    aid: int                           # numeric ID of the assignment (e.g., 1 for A1)
    prereq_ids: Tuple[int, int]        # identifiers of prerequisites (could be input IDs or output IDs)
    output_id: int                     # ID of the output produced by this assignment
    food: str                          # food item required for this assignment
    # Derived data, filled later:
    dependencies: List[int] = field(default_factory=list)  # assignment IDs this assignment depends on

    def __repr__(self) -> str:
        return f"A{self.aid}(food={self.food}, deps={self.dependencies})"


class ScheduleResult:
    """A convenience container to hold schedule results."""

    def __init__(self, strategy: str, schedule: List[List[int]], menus: List[Dict[str, int]],
                 costs_per_day: List[int], total_days: int, total_cost: int, explored_states: Optional[int] = None):
        self.strategy = strategy
        self.schedule = schedule
        self.menus = menus
        self.costs_per_day = costs_per_day
        self.total_days = total_days
        self.total_cost = total_cost
        self.explored_states = explored_states

    def __repr__(self) -> str:
        return (f"ScheduleResult(strategy={self.strategy}, days={self.total_days}, cost={self.total_cost}, "
                f"explored_states={self.explored_states})")


class AssignmentScheduler:
    """
    Core scheduler implementing various greedy strategies and A* search.

    The scheduler operates on a set of `Assignment` objects parsed from an input
    specification. It understands the prerequisite relationships and uses
    different strategies to produce valid day-by-day schedules.
    """

    def __init__(self, costs: Dict[str, int], group_size: int, initial_inputs: Set[int],
                 outputs: Set[int], assignments: List[Assignment]):
        # Validate inputs
        if group_size <= 0:
            raise ValueError("Group size must be positive")
        self.costs = dict(costs)
        self.g = group_size
        self.initial_inputs = set(initial_inputs)
        self.final_outputs = set(outputs)
        # Map output ID to assignment for dependency resolution
        self.assignments: Dict[int, Assignment] = {a.aid: a for a in assignments}
        self.output_to_aid: Dict[int, int] = {a.output_id: a.aid for a in assignments}
        # Build dependencies list for each assignment
        self._resolve_dependencies()
        # Precompute descendant counts for dependency-depth strategy
        self.descendant_counts = self._compute_descendant_counts()
        # Precompute a topological order for earliest-deadline greedy strategy
        self.topo_order = self._compute_topological_order()

    # ------------------------------------------------------------------
    # Parsing and dependency resolution
    # ------------------------------------------------------------------
    def _resolve_dependencies(self) -> None:
        """
        Converts prerequisite identifiers (which could be input IDs or output IDs)
        into actual assignment dependencies. For each assignment, populate
        `dependencies` with IDs of assignments that must be solved before it.
        """
        for assignment in self.assignments.values():
            deps = []
            for pid in assignment.prereq_ids:
                # If pid is available from initial inputs, no assignment dependency
                if pid in self.initial_inputs:
                    continue
                # If pid corresponds to the output of some assignment, add that assignment ID
                if pid in self.output_to_aid:
                    deps.append(self.output_to_aid[pid])
                else:
                    # Unknown prerequisite identifier
                    raise ValueError(f"Unrecognized prerequisite ID {pid} for assignment {assignment.aid}")
            assignment.dependencies = deps

    # ------------------------------------------------------------------
    def _compute_descendant_counts(self) -> Dict[int, int]:
        """
        Computes, for each assignment, the number of downstream assignments reachable
        from it (descendants) in the dependency graph. Used for the dependency-depth
        greedy strategy.
        """
        # Build adjacency list: assignment -> list of children assignments
        adj: Dict[int, List[int]] = {aid: [] for aid in self.assignments}
        for assignment in self.assignments.values():
            for dep in assignment.dependencies:
                # dep -> assignment
                adj[dep].append(assignment.aid)

        # We'll compute descendant counts via DFS with memoization
        visited: Dict[int, int] = {}

        def dfs(aid: int) -> int:
            if aid in visited:
                return visited[aid]
            count = 0
            for child in adj.get(aid, []):
                count += 1 + dfs(child)
            visited[aid] = count
            return count

        for aid in self.assignments:
            dfs(aid)
        return visited

    # ------------------------------------------------------------------
    def _compute_topological_order(self) -> List[int]:
        """
        Computes a topological ordering of assignments based solely on dependency
        relationships. This order is used for the earliest-deadline greedy
        strategy. It ignores group size and food-cost considerations.
        """
        # Kahn's algorithm
        indeg: Dict[int, int] = {aid: 0 for aid in self.assignments}
        for a in self.assignments.values():
            for dep in a.dependencies:
                indeg[a.aid] += 1
        queue = [aid for aid, d in indeg.items() if d == 0]
        topo = []
        while queue:
            # stable order by assignment ID
            queue.sort()
            aid = queue.pop(0)
            topo.append(aid)
            for child in (child for child in self.assignments if aid in self.assignments[child].dependencies):
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)
        return topo

    # ------------------------------------------------------------------
    # Helper for computing available assignments given solved set
    def _available_assignments(self, solved: Set[int]) -> List[int]:
        """
        Returns a list of assignment IDs that have all dependencies satisfied and
        have not yet been solved. The returned list is not sorted; ordering
        strategies apply later.
        """
        avail = []
        for aid, assignment in self.assignments.items():
            if aid in solved:
                continue
            # Check all dependencies
            if all(dep in solved for dep in assignment.dependencies):
                avail.append(aid)
        return avail

    # ------------------------------------------------------------------
    # Greedy strategies

    def _greedy_select(self, avail: List[int], strategy: str, solved: Set[int]) -> List[int]:
        """
        Selects up to g assignments from the list of available assignments
        according to the specified greedy strategy. The order of selection
        influences grouping for that day.

        :param avail: list of available assignment IDs
        :param strategy: one of 'greedy_cost', 'greedy_depth', 'greedy_freq', 'greedy_topo'
        :param solved: set of already solved assignment IDs, used for frequency calculation
        :return: a list of selected assignment IDs for the current day
        """
        # If fewer available assignments than group size, we simply return all
        if not avail:
            return []
        n_take = min(self.g, len(avail))
        # Sort keys based on the chosen heuristic
        if strategy == 'greedy_cost':
            # Sort by ascending cost of their required food; tie-breaker by assignment ID
            sorted_avail = sorted(avail, key=lambda aid: (self.costs[self.assignments[aid].food], aid))
        elif strategy == 'greedy_depth':
            # Sort by descending descendant count (more unlocks first); tie-breaker by assignment ID
            sorted_avail = sorted(avail, key=lambda aid: (-self.descendant_counts.get(aid, 0), aid))
        elif strategy == 'greedy_freq':
            # Compute remaining frequencies of food items among all remaining assignments
            remaining_food_counts: Dict[str, int] = {}
            for rid in self.assignments:
                if rid in solved:
                    continue
                f = self.assignments[rid].food
                remaining_food_counts[f] = remaining_food_counts.get(f, 0) + 1
            # Sort by descending frequency of food item, then by assignment ID
            sorted_avail = sorted(avail,
                                  key=lambda aid: (-remaining_food_counts[self.assignments[aid].food], aid))
        elif strategy == 'greedy_topo':
            # Use precomputed topological order to determine priority
            # Map assignment to its index in topological order; lower index => earlier
            index_map = {aid: idx for idx, aid in enumerate(self.topo_order)}
            sorted_avail = sorted(avail, key=lambda aid: (index_map.get(aid, math.inf), aid))
        else:
            raise ValueError(f"Unknown strategy {strategy}")
        return sorted_avail[:n_take]

    def run_greedy(self, strategy: str) -> ScheduleResult:
        """
        Executes a greedy scheduling algorithm with the specified strategy.

        :param strategy: one of the accepted greedy strategy identifiers
        :return: a ScheduleResult containing the schedule and statistics
        """
        solved: Set[int] = set()
        day = 0
        schedule: List[List[int]] = []
        menus: List[Dict[str, int]] = []
        costs_per_day: List[int] = []
        while len(solved) < len(self.assignments):
            avail = self._available_assignments(solved)
            if not avail:
                # Dead end: no assignments can be selected due to unsatisfied dependencies
                raise RuntimeError("No available assignments to schedule. Check for cycles or unsatisfied dependencies.")
            selected = self._greedy_select(avail, strategy, solved)
            # Record schedule for this day
            schedule.append(selected.copy())
            # Compute menu counts and cost
            menu_counts: Dict[str, int] = {}
            cost_this_day = 0
            for aid in selected:
                food = self.assignments[aid].food
                menu_counts[food] = menu_counts.get(food, 0) + 1
                cost_this_day += self.costs[food]
                # Mark as solved
                solved.add(aid)
            menus.append(menu_counts)
            costs_per_day.append(cost_this_day)
            day += 1
        total_cost = sum(costs_per_day)
        return ScheduleResult(strategy=strategy, schedule=schedule, menus=menus,
                              costs_per_day=costs_per_day, total_days=day, total_cost=total_cost)

    # ------------------------------------------------------------------
    # A* search implementation

    def run_astar(self) -> ScheduleResult:
        """
        Executes an A* search to find the schedule with minimal total food cost.

        The state space consists of subsets of assignments already solved. Each
        successor corresponds to scheduling up to `g` available assignments on a
        new day. The cost of a transition is the sum of food costs for the
        assignments solved on that day. The heuristic estimates the minimal
        remaining cost as the sum of food costs of the unsolved assignments.

        :return: a ScheduleResult with the optimal schedule and statistics.
        """
        total_assignments = len(self.assignments)
        # Represent state by bitmask of solved assignments; mapping assignment ID -> bit index
        aids = sorted(self.assignments.keys())
        aid_to_bit = {aid: i for i, aid in enumerate(aids)}
        all_solved_mask = (1 << total_assignments) - 1

        # Precompute cost of each assignment for quick lookup
        assignment_costs = {aid_to_bit[aid]: self.costs[self.assignments[aid].food] for aid in aids}

        # Precompute prerequisites bitmask for each assignment: bits of assignments that must be solved
        prereq_masks: Dict[int, int] = {}
        for aid in aids:
            bit = aid_to_bit[aid]
            mask = 0
            for dep in self.assignments[aid].dependencies:
                mask |= 1 << aid_to_bit[dep]
            prereq_masks[bit] = mask

        # Precompute heuristic: sum of costs of assignments for quick reference
        total_cost_all = sum(assignment_costs.values())

        # Priority queue items: (f_score, g_cost, solved_mask, day_count, path)
        # path is list of lists of bit indices solved at each day
        pq: List[Tuple[int, int, int, int, List[List[int]]]] = []
        # visited dictionary maps solved_mask to minimal g_cost discovered so far
        visited_cost: Dict[int, int] = {}
        # initial state: no assignments solved
        initial_mask = 0
        heapq.heappush(pq, (total_cost_all, 0, initial_mask, 0, []))
        visited_cost[initial_mask] = 0
        explored_states = 0

        while pq:
            f_score, g_cost, mask, day_count, path = heapq.heappop(pq)
            explored_states += 1
            # If we have solved all assignments, return
            if mask == all_solved_mask:
                # Translate path of bit indices into assignment IDs for each day
                schedule: List[List[int]] = []
                menus: List[Dict[str, int]] = []
                costs_per_day: List[int] = []
                for day_bits in path:
                    assignments_this_day = [aids[bit] for bit in day_bits]
                    schedule.append(assignments_this_day)
                    # compute menu counts and cost
                    menu_counts: Dict[str, int] = {}
                    cost_d = 0
                    for aid in assignments_this_day:
                        food = self.assignments[aid].food
                        menu_counts[food] = menu_counts.get(food, 0) + 1
                        cost_d += self.costs[food]
                    menus.append(menu_counts)
                    costs_per_day.append(cost_d)
                total_days = len(schedule)
                return ScheduleResult(strategy='astar', schedule=schedule, menus=menus,
                                      costs_per_day=costs_per_day, total_days=total_days,
                                      total_cost=g_cost, explored_states=explored_states)

            # Generate available assignments for this state
            # Determine which assignments are available by checking dependencies satisfied
            available_bits: List[int] = []
            for bit in range(total_assignments):
                # skip if already solved
                if mask & (1 << bit):
                    continue
                # prerequisites must be satisfied
                if (prereq_masks[bit] & mask) == prereq_masks[bit]:
                    available_bits.append(bit)
            if not available_bits:
                # no available tasks, skip
                continue
            # Generate combinations of available assignments up to group size
            # We generate combinations of size 1..g of available bits
            max_tasks = min(self.g, len(available_bits))
            # To reduce branching, we can generate combinations sorted by estimated cost ascending (optional)
            # However, we will simply iterate over combinations of size from max_tasks down to 1 to quickly
            # solve more tasks per day (reducing days). This ordering is not essential for correctness but
            # may help to explore solutions with fewer days earlier.
            sizes = range(max_tasks, 0, -1)
            for r in sizes:
                # combinations of r items from available
                for comb in itertools.combinations(available_bits, r):
                    next_mask = mask
                    cost_increment = 0
                    for bit in comb:
                        next_mask |= 1 << bit
                        cost_increment += assignment_costs[bit]
                    new_g_cost = g_cost + cost_increment
                    # heuristic: sum of remaining assignment costs
                    # compute remaining cost quickly via difference of total cost and solved cost
                    # solved cost = g_cost computed so far
                    remaining_cost_est = total_cost_all - new_g_cost
                    # f_score = g + h
                    f_score_next = new_g_cost + remaining_cost_est
                    # If we've seen this state with a lower cost, skip
                    prev_cost = visited_cost.get(next_mask)
                    if prev_cost is not None and prev_cost <= new_g_cost:
                        continue
                    visited_cost[next_mask] = new_g_cost
                    # Append combination to path (store bit indices)
                    new_path = path + [list(comb)]
                    heapq.heappush(pq, (f_score_next, new_g_cost, next_mask, day_count + 1, new_path))
        # If we exit loop without reaching goal, raise error
        raise RuntimeError("A* search failed to find a complete schedule")

    # ------------------------------------------------------------------
    # Utility method to run all greedy strategies
    def run_all_greedies(self) -> List[ScheduleResult]:
        strategies = ['greedy_cost', 'greedy_depth', 'greedy_freq', 'greedy_topo']
        results = []
        for strat in strategies:
            try:
                res = self.run_greedy(strat)
                results.append(res)
            except RuntimeError as e:
                # Skip strategies that fail due to unsatisfiable prerequisites
                results.append(ScheduleResult(strategy=strat, schedule=[], menus=[],
                                              costs_per_day=[], total_days=0, total_cost=math.inf))
        return results


def parse_input_file(path: str) -> Tuple[Dict[str, int], int, Set[int], Set[int], List[Assignment]]:
    """
    Parses an input specification file and returns the scheduling problem data.

    The file format is described in the assignment. Lines beginning with `%` are
    comments. The syntax of relevant lines is:

      C <food-item> <value>     # cost specification
      G <number>                # group size
      I <id> ... -1             # inputs (books/notes) terminated by -1
      O <id> ... -1             # outputs (final outcomes) terminated by -1
      A <id> <input1> <input2> <outcome> <Food-name>  # assignment definition

    :param path: path to the input file
    :return: tuple (costs, group_size, inputs, outputs, assignments)
    """
    costs: Dict[str, int] = {}
    group_size = None
    inputs: Set[int] = set()
    outputs: Set[int] = set()
    assignments: List[Assignment] = []
    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines and comments
            if not stripped or stripped.startswith('%'):
                continue
            parts = stripped.split()
            key = parts[0]
            if key == 'C':
                # C <food-item> <value>
                if len(parts) != 3:
                    raise ValueError(f"Invalid cost specification: {line}")
                food, value = parts[1], int(parts[2])
                costs[food] = value
            elif key == 'G':
                if len(parts) != 2:
                    raise ValueError(f"Invalid group size specification: {line}")
                group_size = int(parts[1])
            elif key == 'I':
                # I <id> ... -1
                for token in parts[1:]:
                    if token == '-1':
                        break
                    inputs.add(int(token))
            elif key == 'O':
                for token in parts[1:]:
                    if token == '-1':
                        break
                    outputs.add(int(token))
            elif key == 'A':
                # A <id> <input1> <input2> <outcome> <Food-name>
                if len(parts) != 6:
                    raise ValueError(f"Invalid assignment specification: {line}")
                aid = int(parts[1])
                prereq1 = int(parts[2])
                prereq2 = int(parts[3])
                out = int(parts[4])
                food = parts[5]
                assignments.append(Assignment(aid=aid, prereq_ids=(prereq1, prereq2), output_id=out, food=food))
            else:
                raise ValueError(f"Unknown line type: {line}")
    if group_size is None:
        raise ValueError("Group size not specified in the input file")
    return costs, group_size, inputs, outputs, assignments


def main():
    parser = argparse.ArgumentParser(description="Assignment scheduling solver")
    parser.add_argument('input_file', help='Path to input specification file')
    parser.add_argument('--strategy', default=None,
                        help=('Strategy to use: greedy_cost, greedy_depth, greedy_freq, '
                              'greedy_topo, astar. If omitted, runs all greedy strategies and astar.'))
    args = parser.parse_args()
    costs, group_size, inputs, outputs, assignments = parse_input_file(args.input_file)
    scheduler = AssignmentScheduler(costs, group_size, inputs, outputs, assignments)
    if args.strategy is None:
        # Run all greedy strategies and A*
        for strat in ['greedy_cost', 'greedy_depth', 'greedy_freq', 'greedy_topo']:
            res = scheduler.run_greedy(strat)
            print(f"Strategy: {strat}")
            for day_idx, day_assignments in enumerate(res.schedule):
                # Convert numeric IDs to A-labels
                assignments_str = ', '.join(f"A{aid}" for aid in day_assignments)
                menu = res.menus[day_idx]
                menu_str = ', '.join(f"{count}-{food}" for food, count in menu.items())
                print(f"Day-{day_idx+1}: {assignments_str}   Menu: {menu_str}   Cost: {res.costs_per_day[day_idx]}")
            print(f"Total Days: {res.total_days}   Total Cost: {res.total_cost}\n")
        # Run A*
        astar_res = scheduler.run_astar()
        print("Strategy: A* (optimal)")
        for day_idx, day_assignments in enumerate(astar_res.schedule):
            assignments_str = ', '.join(f"A{aid}" for aid in day_assignments)
            menu = astar_res.menus[day_idx]
            menu_str = ', '.join(f"{count}-{food}" for food, count in menu.items())
            print(f"Day-{day_idx+1}: {assignments_str}   Menu: {menu_str}   Cost: {astar_res.costs_per_day[day_idx]}")
        print(f"Total Days: {astar_res.total_days}   Total Cost: {astar_res.total_cost}")
        print(f"States explored: {astar_res.explored_states}")
    else:
        if args.strategy == 'astar':
            res = scheduler.run_astar()
        else:
            res = scheduler.run_greedy(args.strategy)
        print(f"Strategy: {res.strategy}")
        for day_idx, day_assignments in enumerate(res.schedule):
            assignments_str = ', '.join(f"A{aid}" for aid in day_assignments)
            menu = res.menus[day_idx]
            menu_str = ', '.join(f"{count}-{food}" for food, count in menu.items())
            print(f"Day-{day_idx+1}: {assignments_str}   Menu: {menu_str}   Cost: {res.costs_per_day[day_idx]}")
        print(f"Total Days: {res.total_days}   Total Cost: {res.total_cost}")
        if res.explored_states is not None:
            print(f"States explored: {res.explored_states}")


if __name__ == '__main__':
    main()
