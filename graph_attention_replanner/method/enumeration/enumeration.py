import os
import csv
import numpy as np
import time
import permutation
from tqdm import tqdm


class Enumeration:
    def __init__(
        self,
        initial_drone_locations,
        task_locations,
        task_times,
        home_depot,
        num_task,
        discretize_level,
        log_file,
        permu_cache_dir,
    ):
        self.initial_drone_locations = initial_drone_locations
        self.task_locations = task_locations
        self.home_depot = home_depot
        self.task_times = task_times
        self.drone_velocity = 1
        self.num_agent = initial_drone_locations.shape[0]
        self.num_node = task_times.shape[0]
        self.num_task = num_task
        self.discretize_level = discretize_level
        self.log_file = log_file
        self.permu_cache_dir = permu_cache_dir

        self.distances = self.generate_distance_matrix(
            initial_drone_locations, task_locations
        )
        self.distances_home_depot = self.generate_distance_matrix_home_depot(
            home_depot, initial_drone_locations, task_locations
        )

    def run(self):
        timer = TicToc()
        timer.tic()
        (
            self.mission_times,
            self.overall_optim_time,
            self.overall_optim_plan,
            self.overall_worst_time,
            self.overall_worst_plan,
            self.exp_count,
        ) = self.compute_paths()

        print("Finished running enumeration.")
        self.wall_time = timer.toc()

        return self.overall_optim_time, self.mission_times, self.overall_optim_plan

    def compute_paths(self):
        print(f"Number of agents: {self.num_agent}")
        print(f"Number of nodes: {self.num_node}")
        print(f"Number of tasks: {self.num_task}")
        print(f"Discretize level: {self.discretize_level}")
        mission_blueprints = permutation.generate_bool_table(
            self.num_agent,
            self.num_node,
            collab=False,
            num_task=self.num_task,
            discretize_level=self.discretize_level,
        )
        num_mission_blueprints = mission_blueprints.shape[0]
        print(f"Number of mission blueprints: {num_mission_blueprints}")

        # Array storing all of the conservative mission times of each permutation
        all_mission_times = []
        overall_optim_time = np.infty
        overall_optim_plan = {}
        overall_worst_time = 0
        overall_worst_plan = {}
        exp_count = 0
        log_feq = 2000  # Log every 2000 bool table

        for mission_batch, mission_blueprint in tqdm(
            enumerate(mission_blueprints),
            desc="Enumerating",
            total=num_mission_blueprints,
        ):
            # Caching permuations
            if os.path.isfile(
                f"{self.permu_cache_dir}/mission_blueprint_{mission_batch}.npy"
            ):
                mission_plans = np.load(
                    f"{self.permu_cache_dir}/mission_blueprint_{mission_batch}.npy"
                )
            else:
                mission_plans = permutation.convert_bool_to_time_table(
                    mission_blueprint
                )
                np.save(
                    f"{self.permu_cache_dir}/mission_blueprint_{mission_batch}.npy",
                    mission_plans,
                )

            num_plan = mission_plans.shape[0]
            mission_time_logs = []

            for mission_plan in mission_plans:
                agent = 0
                agent_times = []
                for agent_path in mission_plan:
                    agent_path = np.array(agent_path)
                    agent_time = 0
                    last_task_in_order = max(
                        agent_path
                    )  # highest number (last) task in the queue of a drone
                    task_order = (
                        1  # integer keeping track of which priority order we are on
                    )
                    prev_task = agent
                    while task_order <= last_task_in_order:
                        current_task = np.where(agent_path == task_order)[0][0] + 1
                        agent_time += self.distances[
                            prev_task, current_task + self.num_agent - 1
                        ]  # travel time
                        task_time = self.task_times[current_task - 1]
                        agent_time += task_time  # task time
                        task_order += 1
                        prev_task = current_task + self.num_agent - 1

                    # Return to home depot, if drone not being used, still go back to home depot
                    agent_time += self.distances_home_depot[0, prev_task]

                    agent_times.append(agent_time)
                    agent += 1

                # Take the max time between all agents to compute conservative mission time
                max_time = max(agent_times)

                if max_time <= overall_optim_time:
                    overall_optim_time = max_time
                    if max_time not in overall_optim_plan:
                        overall_optim_plan[max_time] = [mission_plan]
                    else:
                        overall_optim_plan[max_time].append(mission_plan)
                if max_time >= overall_worst_time:
                    overall_worst_time = max_time
                    if max_time not in overall_worst_plan:
                        overall_worst_plan[max_time] = [mission_plan]
                    else:
                        overall_worst_plan[max_time].append(mission_plan)
                mission_time_logs.append(max_time)

            # Log by batch
            new_exp_count = exp_count + num_plan

            if mission_batch % log_feq == 0:
                # Log now! Must log the last one!
                num_optim_plan = np.array(overall_optim_plan[overall_optim_time]).shape[
                    0
                ]
                num_worst_plan = np.array(overall_worst_plan[overall_worst_time]).shape[
                    0
                ]

                optim_plans = self.numpy_arrays_to_nested_list(
                    overall_optim_plan[overall_optim_time]
                )
                worst_plans = self.numpy_arrays_to_nested_list(
                    overall_worst_plan[overall_worst_time]
                )
                with open(self.log_file, "a") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            new_exp_count,
                            overall_optim_time,
                            num_optim_plan,
                            overall_worst_time,
                            num_worst_plan,
                            optim_plans,
                            worst_plans,
                        ]
                    )

            exp_count = new_exp_count
            all_mission_times.extend(mission_time_logs)

        optim_plans = self.numpy_arrays_to_nested_list(
            overall_optim_plan[overall_optim_time]
        )
        worst_plans = self.numpy_arrays_to_nested_list(
            overall_worst_plan[overall_worst_time]
        )
        return (
            all_mission_times,
            overall_optim_time,
            optim_plans,
            overall_worst_time,
            worst_plans,
            exp_count,
        )

    @staticmethod
    def numpy_arrays_to_nested_list(array_list):
        return [arr.tolist() for arr in array_list]

    def generate_distance_matrix(
        self, initial_drone_locations, task_locations, drone_velocity=1
    ):
        """
        Generates a distance matrix from each waypoint to each other waypoint

        |   | A | B | C | T1 | T2 | T3 | T4 |
        |---|---|---|---|----|----|----|----|
        | A |   |   |   |    |    |    |    |
        | B |   |   |   |    |    |    |    |
        | C |   |   |   |    |    |    |    |
        | T1|   |   |   |    |    |    |    |
        | T2|   |   |   |    |    |    |    |
        | T3|   |   |   |    |    |    |    |
        | T4|   |   |   |    |    |    |    |

        """

        # Create vector of x-locations and y-locations (drones)
        X = [drone_location[0] for drone_location in initial_drone_locations]
        Y = [drone_location[1] for drone_location in initial_drone_locations]

        # Create vector of x-locations and y-locations (tasks)
        X_t = [task[0] for task in task_locations]
        Y_t = [task[1] for task in task_locations]

        # Append
        X = X + X_t
        Y = Y + Y_t

        # Create matrices for each dimension of the problem
        Xx = np.tile(X, (len(initial_drone_locations) + len(task_locations), 1))
        Yy = np.tile(Y, (len(initial_drone_locations) + len(task_locations), 1))

        # Compute distance between each location to each other location
        # Note: diag should be zeros, upper right triangle should be distance from row i to column j
        distance_matrix = (
            np.sqrt((Xx - Xx.T) ** 2 + (Yy - Yy.T) ** 2) / drone_velocity
        )  # [m]

        return distance_matrix

    def generate_distance_matrix_home_depot(
        self, home_depot, initial_drone_locations, task_locations, drone_velocity=1
    ):
        """
        Generates a distance matrix from each waypoint to each other waypoint

        |   | A | B | C | T1 | T2 | T3 | T4 |
        |---|---|---|---|----|----|----|----|
        | A |   |   |   |    |    |    |    |
        | B |   |   |   |    |    |    |    |
        | C |   |   |   |    |    |    |    |
        | T1|   |   |   |    |    |    |    |
        | T2|   |   |   |    |    |    |    |
        | T3|   |   |   |    |    |    |    |
        | T4|   |   |   |    |    |    |    |

        """
        # Reshape home_depot and task_locations for broadcasting
        home_depot = np.array(home_depot)
        task_locations = np.concatenate(
            (np.array(initial_drone_locations), np.array(task_locations))
        )

        # Compute squared differences
        diffs = home_depot[:, np.newaxis, :] - task_locations[np.newaxis, :, :]

        # Sum of squared differences across the x and y coordinates
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        dists = dists / drone_velocity
        return dists

class TicToc:
    def __init__(self):
        self.start_time = None

    def tic(self):
        # print("Timer started.")
        self.start_time = time.time()

    def toc(self):
        if self.start_time is None:
            print("Timer has not been started. Use tic() to start the timer.")
            return None
        elapsed_time = time.time() - self.start_time
        # print(f"Elapsed time: {elapsed_time:.6f} seconds.")
        return elapsed_time