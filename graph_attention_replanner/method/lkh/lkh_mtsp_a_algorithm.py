import lkh
import time
import numpy as np

"""
For asymmetric problems, the solution is 1-indexed and graph is 0-indexed.
For symmetric problems, the solution is 1-indexed and graph is 1-indexed.
Verified with https://profs.info.uaic.ro/mihaela.breaban/mtsplib/MinMaxMTSP/index.html
For data without start loc, pad solution with depot at the start and end of each tour.

For TSP, graph structure cropped edge weight to int, recalculate tour cost
"""


def solve(
    mtsp_problem_type=5,
    salesmen=2,
    filepath="../../../../Downloads/mTSP/INSTANCES/TSP/eil51.tsp",
    solver_path="../../../../LKH-3.0.10/LKH",
):
    with open(filepath, "r") as f:
        problem_str = f.read()
    problem = lkh.LKHProblem.parse(problem_str)

    if problem.type == "TSP":
        G = problem.get_graph()
        coords = [coord[1] for coord in G.nodes(data="coord")]
        distance_matrix = get_distance_matrix(np.array(coords))

    start_time = time.time()
    solution = lkh.solve(
        solver_path,
        problem=problem,
        # Extra parameters
        salesmen=salesmen,
        depot=1,
        mtsp_objective="MINMAX",
    )
    end_time = time.time()
    runtime = end_time - start_time

    def subtract_one_from_nested_list(lst):
        if isinstance(lst, list):
            return [subtract_one_from_nested_list(x) for x in lst]
        else:
            return lst - 1

    def pad_solution_with_depot(tour, depot):
        for i in range(len(tour)):
            tour[i].insert(0, depot)
            tour[i].append(depot)
        return tour

    if problem.type == "ATSP":
        solution = subtract_one_from_nested_list(solution)
        if mtsp_problem_type == 1 or mtsp_problem_type == 2:
            solution = pad_solution_with_depot(solution, 0)
    else:
        if mtsp_problem_type == 1 or mtsp_problem_type == 2:
            solution = pad_solution_with_depot(solution, 1)
    total_weight = []
    for person in range(len(solution)):
        tour = solution[person]
        weight = 0
        for c in range(len(tour) - 1):
            edge = tour[c], tour[c + 1]

            if problem.type == "TSP":
                w = get_edge_weight_for_tsp(edge, distance_matrix)
            else:
                w = problem.get_weight(*edge)

            weight += w
        total_weight.append(weight)
    mission_time = max(total_weight)

    return solution, mission_time, runtime


def get_distance_matrix(coordinates):
    coords = np.array(coordinates)
    return np.sqrt(
        np.sum((coords[:, np.newaxis] - coords[np.newaxis, :]) ** 2, axis=2)
    ).T


def get_edge_weight_for_tsp(edge, distance_matrix):
    return distance_matrix[edge[0] - 1, edge[1] - 1]


# solve(mtsp_problem_type=5, salesmen=2, filepath="../../../../Downloads/mTSP/INSTANCES/TSP/eil51.tsp")
