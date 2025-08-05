from nicegui import events, ui
from bs4 import BeautifulSoup
import numpy as np
import torch
from rl4co.models import REINFORCE
from tensordict.tensordict import TensorDict
import plotly.graph_objects as go
import math
import time

from graph_attention_replanner.config import (
    LogFileConfig,
    get_generator,
    get_env,
)

# Picture credit:
# map: https://www.instructables.com/How-to-Make-a-CUSTOM-Cartoon-MAP-for-FREE/

NUM_NODE = 24
MAX_NUM_AGENT = 6
HOME_DEPOT_LOC = [8, 8]
MAX_TASK_TIME = 10
MIN_TASK_TIME = 2
DT = 0.5
MAP_BG = "media/map_trans.png"
DRONE_ICON = "media/drone.png"

agents = []
agents_id = []
tasks = []
task_times = []
tasks_id = []
state = "Agent"
id_counter = 0


class MissionPlot:
    def __init__(self, mission_details=None):
        self.fig = go.Figure(
            layout=dict(
                xaxis=dict(range=[0, 10]),  # Set x-axis range
                yaxis=dict(range=[10, 0]),  # Set y-axis range & reverse it
                width=380,
                height=380,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                showlegend=False,
            )
        )
        self.ifig = go.Figure(
            layout=dict(
                xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False),
                yaxis=dict(range=[10, 0], showgrid=False, showticklabels=False),
                width=200,
                height=200,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                showlegend=False,
            )
        )

        self.rfig = go.Figure(
            layout=dict(
                xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False),
                yaxis=dict(range=[10, 0], showgrid=False, showticklabels=False),
                width=200,
                height=200,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                showlegend=False,
            )
        )
        self.mission_details = mission_details
        self.locs = None
        self.unsplitted_task_times = None
        self.actions = None
        self.mission_time = None
        self.agent_colors = [
            "blue",
            "orange",
            "red",
            "green",
            "purple",
            "orange",
            "yellow",
        ]
        self.step_counter = 0
        self.max_steps = None
        self.current_agent = 0
        self.current_agent_step = [0] * len(agents)
        self.agent_path_count = [0] * len(agents)
        self.mission_time = None
        self.video = ui.timer(0.1, self.step, active=False)
        self.mp_status = "Initial Planning"
        self.agent_state = []
        self.real_time_traces = []
        self.trace2state_map = []

        with ui.row().classes("w-full justify-start gap-0"):
            with ui.column().classes("w-[50%] h-full items-center"):
                ui.label("Mission In Progress").classes("text-xl text-left font-bold")
                self.plot = ui.plotly(self.fig)

            with ui.column().classes("w-[50%] h-full"):
                # Logo row with fixed height
                with ui.row().classes("w-full h-[45%] items-right"):
                    ui.image("media/logo.png").classes("max-h-full object-contain")

                # ui.space().classes("h-1")

                # Info and controls row with flexible height
                with ui.column().classes("w-full h-[40%] justify-start"):
                    self.mt_textbox = ui.label("Mission Time: NA").classes(
                        "w-full text-left text-lg"
                    )
                    self.mt_possible_mission = ui.label(
                        "Number of Possible Missions: NA"
                    ).classes("w-full text-left text-lg")
                    self.mt_time_taken = ui.label("Time Taken to Plan: NA").classes(
                        "w-full text-left text-lg"
                    )

                    # Group play button and slider horizontally
                    with ui.row().classes("w-full justify-start"):
                        self.play_button = ui.checkbox(
                            "Pause", on_change=self.pause
                        ).bind_value_to(self.video, "active")
                        self.play_button.visible = False
                        ui.label("Progress:").classes(
                            "shrink-0 text-left text-lg"
                        )  # Prevents the label from shrinking
                        self.slider = ui.slider(min=0, max=1, value=0).classes(
                            "flex-1"
                        )  # Makes the slider take up remaining space

    def pause(self, e):
        if e.value:  # Play button is clicked
            return
        if (
            not e.value and self.mp_status == "Initial Planning"
        ):  # Pause button is clicked
            return
        ui.notify(f"Paused at Step: {self.step_counter}/{self.max_steps}")
        print(f"Paused at Step: {self.step_counter}/{self.max_steps}")
        # paused_time = self.mission_time * (self.step_counter-1) / self.max_steps
        paused_time = DT * (self.step_counter - 1)
        self.iplot_mt.text += f"\nPaused at {paused_time:.2f}s"
        # print(f"Paused at {paused_time:.2f}s")
        short_trace, remaining_task_time = self.get_agent_trace(
            self.locs, self.task_times, self.actions, paused_time
        )
        print("Remaining Task Time(Cached):", remaining_task_time)
        remaining_task_time = self.get_combined_task_time(
            remaining_task_time, self.num_task, self.discretize_level
        )
        print("Combined Remaining Task Time(Cached):", remaining_task_time)

        non_zero_indices = [i for i, x in enumerate(remaining_task_time) if x > 0]
        agent_loc = self.get_agent_loc(self.real_time_traces, (self.step_counter - 1))
        # agent_loc = self.get_agent_loc(short_trace)
        remaining_task_time_with_depot = remaining_task_time[non_zero_indices]
        print("Remaining Task Time:", remaining_task_time_with_depot)
        print("Agent Location:", agent_loc)
        print("Task Locations:", self.locs[non_zero_indices])

        # Start Replanning
        self.mission_details.rebuild_mission(
            agent_loc, self.locs[non_zero_indices], remaining_task_time_with_depot
        )

    def get_agent_loc(self, trace, step):
        agent_loc = []
        for agent_idx in range(len(trace)):
            idx = min(step, len(trace[agent_idx]) - 1)  # safety
            print(
                "Agent",
                agent_idx,
                "Step",
                idx,
                "step_counter",
                step,
                "trace_len",
                len(trace),
            )
            agent_loc.append(trace[agent_idx][idx])
        return np.array(agent_loc)

    def get_combined_task_time(self, splitted_task_time, num_task, discretize_level):
        splitted_task_time = splitted_task_time[1:]
        task_time = np.zeros(num_task)
        for i in range(num_task):
            for j in range(discretize_level):
                task_time[i] += splitted_task_time[j * num_task + i]
        task_time = np.round(task_time, decimals=1)
        task_time = np.insert(task_time, 0, 0)  # Add 0 tasktime for home depot
        return task_time

    def static_init_plan(self):
        with ui.column().classes("w-full justify-center gap-0"):
            ui.label("Initial Plan").classes("text-xl text-left font-bold")
            with ui.row().classes("w-full justify-center"):
                self.iplot_mt = ui.label("NA")
            with ui.row().classes("w-full justify-center"):
                self.iplot = ui.plotly(self.ifig)

    def static_replan(self):
        with ui.column().classes("w-full justify-center gap-0"):
            ui.label("Replan").classes("text-xl text-left font-bold")
            with ui.row().classes("w-full justify-center"):
                self.rplot_mt = ui.label("NA")
            with ui.row().classes("w-full justify-center"):
                self.rplot = ui.plotly(self.rfig)

    def ensure_one_trailing_zero(self, arr):
        # First, remove all trailing zeros
        while arr and arr[-1] == 0:
            arr.pop()

        # Then add exactly one zero at the end
        arr.append(0)

        return arr

    def filter_actions(self, actions, num_useful_node, num_agent):
        new_actions = [
            [
                x
                for x in sublist
                if not (num_useful_node < x < NUM_NODE + 1)
                and x < 1 + NUM_NODE + num_agent
            ]
            for sublist in actions
        ]
        # Always return to home depot
        for elem in new_actions:
            elem = self.ensure_one_trailing_zero(elem)
        return new_actions

    def get_possible_missions(self, num_task, num_agent, discretize_level):
        return round(
            math.factorial(num_task * discretize_level + num_agent - 1)
            / math.factorial(num_agent - 1)
        )

    def update_data(
        self,
        locs,
        task_times,
        unsplitted_task_times,
        actions,
        mission_time,
        num_agent,
        num_task,
        discretize_level,
        runtime,
    ):
        if self.mp_status == "Initial Planning":
            target_fig = self.ifig
            target_plot = self.iplot
            self.iplot_mt.text = f"{mission_time:.2f}s"
            self.mp_status = "Replanning"
        else:
            target_fig = self.rfig
            target_plot = self.rplot
            self.rplot_mt.text = f"{mission_time:.2f}s"

        self.mission_time = mission_time
        self.locs = locs
        self.num_task = num_task
        self.discretize_level = discretize_level
        self.unsplitted_locs = locs[: num_task + 1]
        self.unsplitted_task_times = unsplitted_task_times
        self.task_times = task_times
        self.cached_task_time = task_times.copy()
        self.mt_textbox.text = f"Mission Time: {mission_time:.2f}s"
        self.mt_time_taken.text = f"Time Taken to Plan: {runtime:.2f}s"
        self.mt_possible_mission.text = f"Number of Possible Missions: {self.get_possible_missions(num_task, num_agent, discretize_level)}"

        split_indices = np.where(actions == 0)[0]
        split_arrays = np.split(actions, split_indices)
        self.actions = [
            list(sub_arr[sub_arr != 0])
            for sub_arr in split_arrays
            if len(sub_arr[sub_arr != 0]) > 0
        ]

        # Remove padding nodes
        self.actions = self.filter_actions(
            self.actions, num_task * discretize_level, num_agent
        )
        # self.max_steps = math.ceil(mission_time / 0.5)
        self.real_time_traces, _ = self.get_agent_trace(
            locs, task_times, self.actions, mission_time
        )
        for i, arr in enumerate(self.real_time_traces):
            print("Agent", i, "Path", len(arr), "Arr", arr[-10:])

        print("max_steps", self.max_steps)
        print("max step in theory", math.ceil(mission_time / DT))

        # Initial scatter plot for agents
        for agent, nodes in enumerate(self.actions):
            path = locs[nodes]
            print("Agent", agent, "Nodes", nodes)
            self.fig.add_trace(
                go.Scatter(
                    x=[path[0, 0]],
                    y=[path[0, 1]],
                    mode="markers",
                    marker=dict(
                        color=self.agent_colors[agent % len(self.agent_colors)], size=20
                    ),
                    name=agent,
                )
            )
            target_fig.add_trace(
                go.Scatter(
                    x=[path[0, 0]],
                    y=[path[0, 1]],
                    mode="markers",
                    marker=dict(
                        color=self.agent_colors[agent % len(self.agent_colors)], size=20
                    ),
                    name=agent,
                )
            )

            # Add arrow annotations to show direction of movement
            for i in range(len(path) - 1):
                target_fig.add_annotation(
                    x=path[i + 1, 0],
                    y=path[i + 1, 1],
                    ax=path[i, 0],
                    ay=path[i, 1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=1.5,
                    arrowcolor=self.agent_colors[agent % len(self.agent_colors)],
                )

        # Add static nodes
        self.fig.add_trace(
            go.Scatter(
                x=locs[:, 0],
                y=locs[:, 1],
                mode="markers",
                marker=dict(color="black", size=6),
                name="Nodes",
            )
        )
        target_fig.add_trace(
            go.Scatter(
                x=locs[:, 0],
                y=locs[:, 1],
                mode="markers",
                marker=dict(color="black", size=6),
                name="Nodes",
            )
        )

        self.plot.update()
        target_plot.update()
        self.agent_path_count = [0] * num_agent
        self.add_house()

    def add_house(
        self,
    ):
        for f in [self.fig, self.ifig, self.rfig]:
            # Add roof using a triangle (polygon)
            f.add_shape(
                type="path",
                path=f"M {HOME_DEPOT_LOC[0] - 0.35},{HOME_DEPOT_LOC[1] - 0.2} "  # Left point
                f"L {HOME_DEPOT_LOC[0]},{HOME_DEPOT_LOC[1] - 0.5} "  # Top point
                f"L {HOME_DEPOT_LOC[0] + 0.35},{HOME_DEPOT_LOC[1] - 0.2} Z",  # Right point
                fillcolor="black",
                line=dict(color="black"),
            )
            # Add the body of the house using a rectangle
            f.add_shape(
                type="rect",
                x0=HOME_DEPOT_LOC[0] - 0.25,
                y0=HOME_DEPOT_LOC[1] - 0.2,
                x1=HOME_DEPOT_LOC[0] + 0.25,
                y1=HOME_DEPOT_LOC[1] + 0.2,
                line=dict(color="black"),
                fillcolor="black",
            )
        for p in [self.plot, self.iplot, self.rplot]:
            p.update()

    def custom_rounding(self, x, base=0.5):
        return round(x * (1 / base)) * base

    def get_agent_trace(self, locs, task_times, actions, max_time):
        # Create agent_state by inserting -1 between nodes
        agent_state = []
        for action in actions:
            state = []
            for i, node in enumerate(action):
                state.append(node)
                if i < len(action) - 1:
                    state.append(-1)
            agent_state.append(state)

        # Parameters
        speed = 1.0  # m/s
        dt = DT  # time step

        # Initialize state tracking
        print("Initial Task Times", task_times[:9])
        print("Loc", locs[:9])
        n_agents = len(actions)
        positions = [locs[actions[i][0]].tolist() for i in range(n_agents)]
        current_node = [0] * n_agents
        current_task_time = [task_times[actions[i][0]] for i in range(n_agents)]
        self.cached_task_time = task_times.copy()
        paths = [[positions[i]] for i in range(n_agents)]

        rounded_max_time = self.custom_rounding(max_time, base=dt)
        # Simulation loop
        for agent_idx in range(n_agents):
            time = 0
            while time < rounded_max_time:
                if current_task_time[agent_idx] > 0:  # Doing Task
                    print(
                        f"[DEBUG] t={time}/{rounded_max_time} Agent {agent_idx} Doing Task: {actions[agent_idx][current_node[agent_idx]]} Remaining Task Time: {current_task_time[agent_idx]}"
                    )
                    current_task_time[agent_idx] -= dt
                    self.cached_task_time[
                        actions[agent_idx][current_node[agent_idx]]
                    ] -= dt
                    positions[agent_idx] = locs[
                        actions[agent_idx][current_node[agent_idx]]
                    ].tolist()
                    time += dt
                elif (
                    current_node[agent_idx] < len(actions[agent_idx]) - 1
                ):  # Target Node Exist
                    start = locs[actions[agent_idx][current_node[agent_idx]]]
                    end = locs[actions[agent_idx][current_node[agent_idx] + 1]]
                    direction = end - start
                    distance = np.linalg.norm(direction)
                    if distance <= 0:
                        print(
                            f"[DEBUG] t={time}/{rounded_max_time} Agent {agent_idx} Do Nth"
                        )
                        # Dont count time, Dont add pos to trace
                        current_node[agent_idx] += 1
                        current_task_time[agent_idx] = task_times[
                            actions[agent_idx][current_node[agent_idx]]
                        ]
                    else:
                        # Traversing
                        step = speed * dt / distance * direction
                        new_pos = positions[agent_idx] + step
                        distance_to_task = np.linalg.norm(new_pos - end)
                        # If close enough to target, snap to target and switch to next task
                        print(
                            f"[DEBUG] t={time}/{rounded_max_time} Agent {agent_idx} Traversing To Task: {actions[agent_idx][current_node[agent_idx] + 1]} Distance to task: {distance_to_task} Task Time: {current_task_time[agent_idx]}"
                        )
                        if distance_to_task < speed * dt:
                            new_pos = end
                            current_node[agent_idx] += 1
                            current_task_time[agent_idx] = task_times[
                                actions[agent_idx][current_node[agent_idx]]
                            ]
                        positions[agent_idx] = new_pos.tolist()
                        time += dt
                else:
                    print(
                        f"[DEBUG] t={time}/{rounded_max_time} Agent {agent_idx} Finished All Tasks"
                    )
                    # positions[agent_idx] = locs[actions[agent_idx][current_node[agent_idx]]].tolist() # Stay at last node
                    positions[agent_idx] = None
                    time += dt

                if positions[agent_idx]:
                    paths[agent_idx].append(positions[agent_idx])

        max_steps = 0
        for agent in range(n_agents):
            max_steps = max(max_steps, len(paths[agent]))
        self.max_steps = max_steps
        print("Max Steps", max_steps)

        return paths, self.cached_task_time

    def step(self):
        if self.step_counter >= self.max_steps:
            self.video.deactivate()
            print("Video Ended")
            return

        self.slider.set_value((self.slider.value + 1 / self.max_steps) % 1)
        # print(f"Step {self.step_counter}/{self.max_steps}")
        self.fig.data = []
        self.fig.add_trace(
            go.Scatter(
                x=self.locs[:, 0],
                y=self.locs[:, 1],
                mode="markers",
                marker=dict(color="black", size=6),
                name="Nodes",
            )
        )

        for agent in range(len(self.real_time_traces)):
            # Plot path history up to current step
            for i in range(1, self.step_counter + 1):
                if i < len(self.real_time_traces[agent]):
                    prev_pos = self.real_time_traces[agent][i - 1]
                    curr_pos = self.real_time_traces[agent][i]
                    self.fig.add_trace(
                        go.Scatter(
                            x=[prev_pos[0], curr_pos[0]],
                            y=[prev_pos[1], curr_pos[1]],
                            mode="lines",
                            line=dict(
                                color=self.agent_colors[agent % len(self.agent_colors)],
                                width=2,
                            ),
                            name=f"Agent {agent} Path",
                        )
                    )

            # Plot the agent's current position
            agent_step = min(self.step_counter, len(self.real_time_traces[agent]) - 1)
            agent_pos = self.real_time_traces[agent][agent_step]
            self.fig.add_trace(
                go.Scatter(
                    x=[agent_pos[0]],
                    y=[agent_pos[1]],
                    mode="markers",
                    marker=dict(
                        color=self.agent_colors[agent % len(self.agent_colors)], size=20
                    ),
                    name=f"Agent {agent}",
                )
            )

            self.agent_path_count[agent] += 1

        self.fig.add_trace(
            go.Scatter(
                x=self.locs[:, 0],
                y=self.locs[:, 1],
                mode="markers",
                marker=dict(color="black", size=6),
                name="Nodes",  # Set a single trace name
            )
        )
        self.plot.update()
        self.step_counter += 1

    def clear(self):
        self.fig.data = []
        self.plot.update()
        self.ifig.data = []
        self.ifig.layout.annotations = []
        self.iplot.update()
        self.rfig.data = []
        self.rfig.layout.annotations = []
        self.rplot.update()

        self.slider.set_value(0)
        self.mt_textbox.text = "Mission Time: NA"
        self.mt_possible_mission.text = "Number of Possible Missions: NA"
        self.mt_time_taken.text = "Time Taken to Plan: NA"
        self.rplot_mt.text = "NA"
        self.iplot_mt.text = "NA"
        self.video.deactivate()
        self.play_button.set_value(False)

        self.locs = None
        self.actions = None
        self.mission_time = None
        self.unsplitted_locs = None
        self.unsplitted_task_times = None
        self.agent_state = []
        self.real_time_traces = []
        self.trace2state_map = []
        self.step_counter = 0
        self.current_agent = 0
        self.current_agent_step = [0] * len(agents)
        self.agent_path_count = [0] * len(agents)
        self.add_house()

    def clear_part(self):
        self.fig.data = []
        self.plot.update()
        self.rfig.data = []
        self.rfig.layout.annotations = []
        self.rplot.update()

        self.slider.set_value(0)
        self.mt_textbox.text = "Mission Time: NA"
        self.rplot_mt.text = "NA"
        self.video.deactivate()
        self.play_button.set_value(False)

        self.locs = None
        self.actions = None
        self.mission_time = None
        self.unsplitted_locs = None
        self.unsplitted_task_times = None
        self.agent_state = []
        self.real_time_traces = []
        self.trace2state_map = []
        self.step_counter = 0
        self.current_agent = 0
        self.current_agent_step = [0] * len(agents)
        self.agent_path_count = [0] * len(agents)
        self.add_house()


def get_model_cfg():
    # Generalised model
    model_cfg = LogFileConfig(
        5,
        NUM_NODE,
        1,
        6,
        1,
        4,
        1,
        6,
        1,
    )
    return model_cfg


def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg = get_model_cfg()
    model_path = model_cfg.get_train_modelpath(seed=0)
    model = REINFORCE.load_from_checkpoint(
        model_path, load_baseline=False, strict=False
    )
    model = model.to(device)
    return model


def plan():
    global md, mp

    if np.array(md.agents).shape[0] > MAX_NUM_AGENT:
        ui.notify("Too many agents. Please delete some agents and try again")
        return
    if np.array(md.tasks).shape[0] * md.discretize_level > NUM_NODE:
        ui.notify(
            f"Too many node. The product of number of tasks and level of collaboration should be less than {NUM_NODE}. Please delete some tasks or lower the level of collaboration and try again"
        )

    ui.notify("Planning")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    td_data = parse_input_data(md.agents, md.tasks, md.discretize_level).to(device)
    print(td_data)

    if td_data is None:
        ui.notify("No input data")
        return

    mtsp_problem_type = 5
    generator = get_generator(mtsp_problem_type)(
        num_node=NUM_NODE,
        min_num_task=1,
        max_num_task=6,
        min_discretize_level=1,
        max_discretize_level=4,
        min_num_agent=1,
        max_num_agent=6,
        seed=0,
    )
    env = get_env(mtsp_problem_type)(generator)
    td = env.reset(td_data)
    print("start inference")
    start_time = time.time()
    out = model(td, phase="test", decode_type="greedy", return_actions=True)
    end_time = time.time()
    runtime = end_time - start_time
    mission_time = -out["reward"][0]
    actions = out["actions"][0]
    print(f"Mission time: {mission_time}; Actions: {actions}")

    if mp.mp_status == "Initial Planning":
        mp.clear()
    else:
        mp.clear_part()

    mp.update_data(
        locs=td_data["locs"][0].numpy(),
        task_times=td_data["task_length"][0].numpy(),
        unsplitted_task_times=td_data["unsplited_task_length"][0].numpy(),
        actions=actions.numpy(),
        mission_time=mission_time.numpy(),
        num_agent=len(md.agents),
        num_task=len(md.tasks),
        discretize_level=md.discretize_level,
        runtime=runtime,
    )
    mp.play_button.set_value(True)

    return mission_time, actions


def parse_input_data(agents, tasks, discretize_level):
    if not agents or not tasks:
        print(f"Agents {agents} or tasks {tasks} are empty")
        return

    scale_factor = 10 / 800  # Scale coordinates from 800x800 grid to 10x10 grid
    agents = np.round(np.array(agents) * scale_factor, 1)
    tasks = np.array(tasks)
    home_depot = np.array([[HOME_DEPOT_LOC]])

    num_agent = agents.shape[0]
    num_task = tasks.shape[0]

    task_length = np.zeros((1, NUM_NODE + 1 + MAX_NUM_AGENT))
    unsplited_task_length = np.zeros((1, NUM_NODE + 1 + MAX_NUM_AGENT))
    unsplited_task_length[0, 1 : num_task + 1] = tasks[:, 2].reshape(1, num_task)
    task_divided = tasks[:, 2].reshape(1, num_task) / discretize_level
    task_length[0, 1 : num_task * discretize_level + 1] = np.repeat(
        task_divided, discretize_level, axis=0
    ).reshape(1, num_task * discretize_level)

    # print("[DEBUG]", task_length.shape)
    discretize_levels_distribution = np.ones((1)) * discretize_level

    start_locs = agents.reshape(1, num_agent, 2)
    task_original = tasks[:, :2].reshape(1, num_task, 2) * scale_factor
    task_original = np.round(task_original, 1)
    print("[DEBUG] task_original", task_original)

    locs = np.repeat(task_original, discretize_level, axis=0).reshape(
        1, num_task * discretize_level, 2
    )
    print("[DEBUG] locs", locs)

    # Add home depot to locs then pad up to 24
    locs = np.concatenate((home_depot, locs), axis=1)
    pad_width = [
        (0, 0),
        (0, NUM_NODE + 1 - locs.shape[1]),
        (0, 0),
    ]  # (1st dim: no change, 2nd dim: pad after, 3rd dim: no change)
    locs = np.pad(locs, pad_width, mode="constant", constant_values=0)
    # Add start locs to locs then pad up to 1+24+max_num_agent
    locs = np.concatenate((locs, start_locs), axis=1)
    pad_width = [
        (0, 0),
        (0, NUM_NODE + 1 + MAX_NUM_AGENT - locs.shape[1]),
        (0, 0),
    ]  # (1st dim: no change, 2nd dim: pad after, 3rd dim: no change)
    locs = np.pad(locs, pad_width, mode="constant", constant_values=0)

    num_agent_for_td = np.array(num_agent).reshape(1)
    num_task_for_td = np.array(num_task * discretize_level).reshape(1)

    td = TensorDict(
        {
            "locs": torch.Tensor(locs),
            "num_task": torch.Tensor(num_task_for_td),
            "num_agent": torch.Tensor(num_agent_for_td),
            "task_length": torch.Tensor(task_length),
            "unsplited_task_length": torch.Tensor(unsplited_task_length),
            "discretize_levels_distribution": torch.Tensor(
                discretize_levels_distribution
            ),
            "start_locs": torch.Tensor(start_locs),
        },
        batch_size=1,
    )
    return td


class MissionDetails:
    def __init__(self):
        self.agents = []
        self.agents_id = []
        self.tasks = []
        self.tasks_id = []
        self.selected_tasks = []
        self.task_times = np.random.choice(
            range(MIN_TASK_TIME, MAX_TASK_TIME), size=NUM_NODE
        )
        self.state = "Agent"
        self.id_counter = 0
        self.discretize_level = 1

    def find_element_type(self, id):
        if id in self.agents_id:
            return "Agent"
        elif id in self.tasks_id:
            return "Task"
        else:
            return None

    def set_discretize_level(
        self,
    ):
        self.discretize_level = dl_slider.value
        ui.notify(f"Discretize Level: {self.discretize_level}")

    def set_to_draw_agent(self):
        self.state = "Agent"

    def set_to_draw_task(self):
        self.state = "Task"

    def add_agent(self, x, y):
        self.agents.append([x, y])
        id = self.id_counter
        self.agents_id.append(id)
        self.id_counter += 1

        color = "SteelBlue"
        ii.content += f'<circle cx="{x}" cy="{y}" id ="{id}" r="30" fill="none" stroke="{color}" stroke-width="4" pointer-events="all" cursor="pointer" />'
        return id

    def add_task(self, x, y):
        id = self.id_counter
        self.tasks_id.append(id)
        task_time_idx = len(self.tasks_id) - 1
        task_time = self.task_times[task_time_idx]
        self.tasks.append([x, y, task_time])
        self.id_counter += 1

        color = "Orange"
        ii.content += f'<circle cx="{x}" cy="{y}" id ="{id}" r="30" fill="{color}" stroke="{color}" stroke-width="4" pointer-events="all" cursor="pointer" />'
        ii.content += f'<text x="{x-7}" y="{y+7}" id ="{id}" font-size="24" fill="white">{task_time}</text>'
        return id, task_time

    def delete_element(self, id, type="Agent"):
        if type == "Agent":
            idx = self.agents_id.index(id)
            self.agents_id.pop(idx)
            self.agents.pop(idx)
        else:
            idx = self.tasks_id.index(id)
            self.tasks_id.pop(idx)
            self.tasks.pop(idx)

    def add_selected_task(self, id):
        self.selected_tasks.append(id)

    def delete_all_selected_task(self):
        for id in self.selected_tasks:
            self.delete_element(id, "Task")

        soup = BeautifulSoup(ii.content)
        elements = []
        for id in self.selected_tasks:
            elements += soup.find_all("circle", id=id)
            elements += soup.find_all("text", id=id)
            elements += soup.find_all("rect", id=id)
        for element in elements:
            element.decompose()
        ii.content = str(soup)  # Update the content
        self.selected_tasks = []

    def add_time_to_all_selected_task(self):
        for id in self.selected_tasks:
            idx = self.tasks_id.index(id)
            original_time = self.task_times[idx]
            new_time = original_time + 1
            self.task_times[idx] = min(new_time, MAX_TASK_TIME)  # Cap at 10
            self.tasks[idx][2] = self.task_times[idx]

        self.update_selected_task_visual_for_changed_time()

    def reduce_time_to_all_selected_task(self):
        for id in self.selected_tasks:
            idx = self.tasks_id.index(id)
            original_time = self.task_times[idx]
            new_time = original_time - 1
            self.task_times[idx] = max(new_time, MIN_TASK_TIME)
            self.tasks[idx][2] = self.task_times[idx]

        self.update_selected_task_visual_for_changed_time()

    def update_selected_task_visual_for_changed_time(self):
        soup = BeautifulSoup(ii.content)
        text_elements = []
        for id in self.selected_tasks:
            text_elements += soup.find_all("text", id=id)

        for elem in text_elements:
            id = elem["id"]
            elem.string = str(self.task_times[self.tasks_id.index(int(id))])
        ii.content = str(soup)

    def get_elem_location(self, id):
        if id in self.agents_id:
            idx = self.agents_id.index(id)
            return self.agents[idx]
        elif id in self.tasks_id:
            idx = self.tasks_id.index(id)
            return self.tasks[idx][0:2]
        else:
            print("Element not found")
            return None

    def clear_content(self):
        self.agents = []
        self.agents_id = []
        self.tasks = []
        self.tasks_id = []
        self.selected_tasks = []
        self.id_counter = 0
        self.task_times = np.random.choice(range(1, 11), size=NUM_NODE)

    def build_base_plot(self):
        ii.content = ""
        scale_factor = 800 / 10
        # Add an SVG-based house icon instead of a rectangle
        ii.content += f"""
            <svg x="{HOME_DEPOT_LOC[0] * scale_factor - 30}" 
                y="{HOME_DEPOT_LOC[1] * scale_factor - 30}" 
                width="60" 
                height="60" 
                viewBox="0 0 64 64" 
                xmlns="http://www.w3.org/2000/svg">
                <path d="M32 12L4 40h8v16h16V44h8v12h16V40h8z" 
                    fill="black" 
                    stroke="#000" 
                    stroke-width="2"/>
            </svg>
        """

    def add_element(self, e: events.MouseEventArguments):
        if not e.type == "mousedown":
            return
        if self.state == "Agent":
            self.add_agent(e.image_x, e.image_y)
        else:  # state == 'Task'
            self.add_task(e.image_x, e.image_y)

    def update_mission(self, e: events.MouseEventArguments):
        id = int(e.args["element_id"])
        print(f"Element {id} clicked")
        soup = BeautifulSoup(ii.content)

        boxes = soup.find_all("rect", id=id)
        if len(boxes) > 0:
            ui.notify(f"Unselect Task {id}")
            for box in boxes:
                box.decompose()
            ii.content = str(soup)
            self.selected_tasks.remove(id)
            if len(self.selected_tasks) == 0:
                for button in task_related_buttons:
                    button.set_enabled(False)
            return

        elements = soup.find_all("circle", id=id)
        elements += soup.find_all("text", id=id)
        element_type = self.find_element_type(id)
        if element_type == "Task":
            ui.notify(f"Selected Task {id}")
            self.add_selected_task(id)
            x, y = self.get_elem_location(id)
            ii.content += f'<rect id="{id}" x="{x-40}" y="{y-30}" width="80" height="60" fill="none" stroke="red" stroke-width="4" pointer-events="all" cursor="pointer" />'
            for button in task_related_buttons:
                button.set_enabled(True)

        elif element_type == "Agent":
            # Deleting agents
            ui.notify("Delete Agent")
            for element in elements:
                element.decompose()
            ii.content = str(soup)
            self.delete_element(id, "Agent")
        else:
            ui.notify("There is a problem, please try clear the mission.")

    def beautify_int(self, arr):
        new_arr = []
        for t in arr:
            if t.is_integer():
                new_arr.append(int(t))
            else:
                new_arr.append(np.round(t, 1))
        return new_arr

    def rebuild_mission(self, agents, tasks, task_times):
        print("Rebuilding Mission")
        print("Previous Task Time", self.task_times)
        # Beautify int
        self.clear_content()
        self.build_base_plot()

        # Pad with random values from 1 to 10
        pad_size = NUM_NODE - len(task_times)
        if pad_size > 0:
            padding = np.random.choice(range(1, 11), size=pad_size)
            self.task_times = np.concatenate([task_times, padding])

        self.task_times = self.beautify_int(self.task_times)
        print("New Task Time", self.task_times)

        scale_factor = 800 / 10

        for task in tasks:
            self.add_task(task[0] * scale_factor, task[1] * scale_factor)

        for agent in agents:
            self.add_agent(agent[0] * scale_factor, agent[1] * scale_factor)


"""
-------------------------------------------------------------------------------------------GUI------------------------------------------------------------------------------------------
"""


def clear_plots():
    global md, mp
    md.clear_content()
    md.build_base_plot()
    mp.mp_status = "Initial Planning"
    mp.clear()
    ui.notify("Clear All")


model = load_model()
md = MissionDetails()
with ui.row().classes("w-full h-[95vh] flex"):
    # Left column (50% of screen width)
    with ui.column().classes("w-[35%] h-full"):
        with ui.card().classes("w-full h-full"):
            ui.label("Set Up Mission").classes("text-xl font-bold")

            # Left content
            with ui.row().classes("w-full justify-center"):
                ii = (
                    ui.interactive_image(
                        MAP_BG,
                        size=(800, 800),
                        cross=True,
                        events=["mousedown"],
                        on_mouse=md.add_element,
                    )
                    .classes("w-4/5 bg-blue-50")
                    .on("svg:pointerdown", md.update_mission)
                )
                md.build_base_plot()
            discretize_level_text = ui.label(
                "Level of Collaboration (1 Task = n Sub-Tasks)"
            ).classes("ml-2 text-left text-lg")
            with discretize_level_text:
                md.discretize_level = 1
                dl_slider = ui.slider(min=1, max=4, step=1, value=1).on(
                    "change", md.set_discretize_level
                )
                dl_slider_pos = ui.label()
                dl_slider_pos.bind_text_from(dl_slider, "value")

            with ui.row():
                ui.button("Agent", on_click=md.set_to_draw_agent)
                ui.button("Task", on_click=md.set_to_draw_task)
                ui.button("Clear", on_click=clear_plots)
                ui.button("Plan", on_click=plan)
                ui.button("Pause", on_click=lambda: mp.play_button.set_value(False))

            # Grey out when no task is selected
            with ui.row():
                ui.label("Options for Selected Tasks: ")
            with ui.row():
                task_related_buttons = []
                task_related_buttons.append(
                    ui.button("Delete", on_click=md.delete_all_selected_task)
                )
                task_related_buttons.append(
                    ui.button("+1s", on_click=md.add_time_to_all_selected_task).props(
                        "no-caps"
                    )
                )
                task_related_buttons.append(
                    ui.button(
                        "-1s", on_click=md.reduce_time_to_all_selected_task
                    ).props("no-caps")
                )
                for button in task_related_buttons:
                    button.set_enabled(False)

    # Right column (50% of screen width)
    with ui.column().classes("w-[60%] h-full"):
        # Top card (50% of right column height)
        with ui.card().classes("w-full h-[60%] mb-2"):
            mp = MissionPlot(mission_details=md)

        # Bottom row with two cards (50% of right column height)
        with ui.row().classes("w-full h-[40%]"):
            # Left card of bottom row
            with ui.card().classes("w-[48%] h-full mr-1"):
                # ui.label("Initial Plan").classes("text-xl font-bold")
                mp.static_init_plan()

            # Right card of bottom row
            with ui.card().classes("w-[48%] h-full ml-1"):
                # ui.label("Replan").classes("text-xl font-bold")
                mp.static_replan()
                mp.add_house()

ui.run()
