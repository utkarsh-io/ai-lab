Assignment 4 Scheduling Solver
===============================

This repository contains a Python implementation of the assignment scheduling
problem described in *MA3206: Artificial Intelligence — Assignment 4*.  The
problem involves scheduling a set of interdependent assignments subject to
prerequisite constraints and limited daily capacity.  Each assignment
requires a food item, and the objective is to generate a valid schedule that
minimizes the total cost of these food items.  Both greedy heuristics and
an optimal A* search are implemented.

Files included
--------------

* **scheduler.py** – core module that parses an input specification and
  produces schedules using various greedy strategies or an A* search.
* **test1.txt**, **test2.txt**, **test3.txt** – three sample test cases (each
  containing at least 10 assignments) formatted according to the assignment
  specification.
* **report.pdf** – a report summarising the implemented strategies, results
  obtained on the test cases, and conclusions (generated separately).
* **README.txt** – this file, explaining how to run the program and its
  dependencies.

Dependencies
------------

The scheduling code relies solely on Python’s standard library and has been
tested under Python 3.13.5.  No external libraries are required to run
`scheduler.py`.  The provided report (`report.pdf`) was generated using the
ReportLab library, which is available in the environment supplied.  If you
wish to regenerate the report yourself, ensure that `reportlab` is installed
(`pip install reportlab`).

Running the program
-------------------

To execute the scheduler on a given input file, run:

```bash
python scheduler.py <input_file> [--strategy STRATEGY]
```

Where `STRATEGY` can be one of the following:

* **greedy_cost** – Greedy by food cost.  At each day the scheduler selects
  up to `g` assignments with the lowest food cost, trying to minimise the
  immediate cost of the daily menu.
* **greedy_depth** – Greedy by dependency depth.  This heuristic chooses
  assignments that unlock the largest number of downstream assignments (the
  “critical path” first), allowing more flexibility later.
* **greedy_freq** – Greedy by food type frequency.  Assignments whose
  required food item occurs most frequently among the remaining assignments are
  prioritised.  The intuition is to maximise reuse of the menu on a given
  day.
* **greedy_topo** – Greedy by earliest deadline (topological order).  A
  precomputed topological order of the dependency graph is used to schedule
  assignments as early as possible.
* **astar** – Optimal scheduling via A* search.  This method explores the
  space of all possible schedules and returns the one with minimal total
  food cost.

If `--strategy` is omitted, the program runs all four greedy heuristics and
also the A* search on the provided input file.  Example usage:

```bash
# Run all strategies on test case 1
python scheduler.py test1.txt

# Run only the dependency–depth greedy strategy on test case 2
python scheduler.py test2.txt --strategy greedy_depth

# Run the optimal A* search on test case 3
python scheduler.py test3.txt --strategy astar
```

The program prints a day‑by‑day schedule, the daily menu (food item counts),
and the cost per day.  At the end of the output, it reports the total
number of days and the total cost.  When running A* search, it also
reports the number of states explored before reaching the goal.

Greedy strategies
-----------------

### Greedy by food cost (`greedy_cost`)

At each step, the scheduler considers all assignments whose prerequisites
have been satisfied.  It selects up to `g` assignments with the lowest
associated food cost.  Ties are broken by assignment ID.  This strategy
attempts to minimise the daily menu cost, potentially grouping low‑cost
assignments together.

### Greedy by dependency depth (`greedy_depth`)

For each assignment, the algorithm precomputes how many downstream tasks
(descendants) depend on it.  Assignments with larger descendant counts are
scheduled earlier, under the intuition that completing them sooner will make
more assignments available and reduce idle time later.  At each day, up to
`g` available assignments with the highest descendant counts are selected.

### Greedy by food type frequency (`greedy_freq`)

This heuristic counts how many remaining assignments require each food
item.  Assignments whose food type occurs most frequently are prioritised,
with the hope that grouping such tasks reduces menu diversity in a single
day.  Ties are broken by assignment ID.

### Greedy by earliest deadline / topological order (`greedy_topo`)

A pure topological order of the dependency graph is computed in advance.
Assignments are scheduled strictly according to this order, up to `g` per
day.  This strategy minimises idle days by scheduling tasks as early as
possible without considering food cost.

A* search
---------

The A* algorithm explores the space of partial schedules using a priority
queue.  Each state is defined by the subset of assignments that have been
completed.  Possible actions from a state are combinations of up to `g`
available assignments to execute on the next day.  The cost of a transition
is the sum of food costs for that day’s assignments.  The heuristic used
for A* is admissible: it is the sum of the food costs of the remaining
assignments (ignoring dependencies).  This represents the minimal
additional cost needed to finish all tasks and ensures optimality.

Generating the report
---------------------

The provided `report.pdf` was produced by running the scheduler on the
three test cases and summarising the results.  To regenerate it, you can
use the `report_generator.py` (if supplied) or write your own script using
ReportLab.  The main contents include descriptions of each heuristic, a
statistical comparison table across the test cases, plots comparing total
costs and days, and a discussion of observations.

Contact
-------

For any questions regarding this implementation, please reach out to the
assignment authors.