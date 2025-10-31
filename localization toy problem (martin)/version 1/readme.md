The problem is self-contained within the toy.py file.

There are 3 main data structures/classes to account for:
- Monitoring
  - This is the main data struct that tracks which nodes are monitors, which
    monitoring paths / cycles are in use, and how good they are at providing
    coverage.
  - This data structure contains the bulk of the functions that are used to
    operate tests.
- EStats
  - This structure tracks which monitoring paths / cycles cover which edges as
    well as how many paths / cycles are on an edge.
- MStats
  - This data struct holds all of the information of a monitoring path / cycle.
    The Monitoring data struct is contained as a list in both Estats and the
    parent Monitoring class.
