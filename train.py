import os
import subprocess
import json
import math
import numpy as np
import pandas as pd
import shutil
import gym
import toml  # pip install toml
import matplotlib.pyplot as plt
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from itertools import product
import torch.nn.functional as F  # NEW: for one-hot encoding in the forward pass


# For the NN version:
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Setup logging and seeds
# -----------------------------
logging.basicConfig(level=logging.INFO)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# New coefficients for the mixed reward function
name = "alpha"
ALPHA_INTERVENTION = 1.0  # Weight for intervention cost
BETA_PEAK = 0  # Weight for infection penalty if above threshold
LAMBDA_TERMINAL = 0  # Weight for terminal outcome penalty
I_CRITICAL = 0.02  # Critical infection threshold

OUTPUT_DIR = f"outputs_{name}"
POLICY_DIR = f"policy_{name}"
CONFIG_DIR = f"configs_{name}"
MODEL_DIR = f"model_{name}"
BASE_POLICY_PATH = f"policy_{name}.json"
LOSS_FILE = f"losses_{name}.png"

# Clean up outputs, policy, and config folders
for f in [OUTPUT_DIR, POLICY_DIR, CONFIG_DIR, MODEL_DIR]:
    shutil.rmtree(f, ignore_errors=True)
    os.makedirs(f, exist_ok=True)

os.makedirs(MODEL_DIR, exist_ok=True)

###############################
# Global hyperparameters
###############################

# Simulation / Environment parameters
S_DISCRETIZATION = 0.02  # Ideally read from config.toml
I_DISCRETIZATION = 0.01  # Ideally read from config.toml

# Intervention values
INTERVENTION_VALUES = [0.0, 0.5, 0.75, 0.9]
NUM_ACTIONS = len(INTERVENTION_VALUES) ** 2

# Learning hyperparameters
ALPHA = 0.1  # learning rate
GAMMA = 1.0  # discount factor
NUM_EPISODES = 100  # number of training episodes

# Epsilon
EPSILON = np.linspace(0.1, 0.001, NUM_EPISODES).tolist()

# Command to run the Julia simulation (adjust as needed)
JULIA_CMD = ["julia", "--project=.", "abm/simulation.jl"]
NUM_ADULTS = 0.7

# Base path to config file (assumed to be in the current working directory)
CONFIG_PATH = "config.toml"

I_max = 0.22  # Maximum allowed I value
S_min = 0.24  # Minimum allowed S value

###############################
# Utility functions for discretization and action mapping
###############################


def discretize_state(state, s_disc=S_DISCRETIZATION, i_disc=I_DISCRETIZATION):
    precision_s = len(str(s_disc).split(".")[-1])
    precision_i = len(str(i_disc).split(".")[-1])
    disc_state = [
        round(math.floor(state[0] / s_disc) * s_disc, precision_s),
        round(math.floor(state[1] / s_disc) * s_disc, precision_s),
        round(math.floor(state[2] / i_disc) * i_disc, precision_i),
        round(math.floor(state[3] / i_disc) * i_disc, precision_i),
    ]
    return disc_state


def state_key(state):
    disc_state = discretize_state(state)
    return ",".join(str(x) for x in disc_state)


def index_to_action(idx):
    num_vals = len(INTERVENTION_VALUES)
    w = INTERVENTION_VALUES[idx // num_vals]
    sch = INTERVENTION_VALUES[idx % num_vals]
    return (w, sch)


def action_to_index(action):
    w, sch = action
    try:
        i_w = INTERVENTION_VALUES.index(w)
        i_s = INTERVENTION_VALUES.index(sch)
        return i_w * len(INTERVENTION_VALUES) + i_s
    except ValueError:
        raise ValueError("Action values not in the allowed discrete set.")


###############################
# Policy representation and saving
###############################


def generate_full_policy(policy_table):
    precision_s = len(str(S_DISCRETIZATION).split(".")[-1])
    precision_i = len(str(I_DISCRETIZATION).split(".")[-1])
    s_vals = np.round(
        np.arange(S_min, 1 + S_DISCRETIZATION, S_DISCRETIZATION), precision_s
    )
    i_vals = np.round(
        np.arange(0, I_max + I_DISCRETIZATION, I_DISCRETIZATION), precision_i
    )

    # Optional: Precompute string representations if needed.
    s_str = [str(val) for val in s_vals]
    i_str = [str(val) for val in i_vals]

    # Precompute action mapping for all possible indices.
    action_cache = {i: index_to_action(i) for i in range(NUM_ACTIONS)}

    # Use dictionary comprehension with itertools.product.
    full_policy = {
        f"{s}:{s2}:{i}:{i2}".replace(":", ","): action_cache[
            policy_table.get(f"{s}:{s2}:{i}:{i2}".replace(":", ","), 0)
        ]
        for s, s2, i, i2 in product(s_str, s_str, i_str, i_str)
    }

    return full_policy


def save_policy(policy_table, filename):
    """
    Save the full policy mapping to a JSON file.
    """
    full_policy = generate_full_policy(policy_table)
    with open(filename, "w") as f:
        json.dump(full_policy, f, indent=4)


###############################
# Gym Environment that wraps the simulation
###############################


class SimulationEnv(gym.Env):
    """
    A custom Gym environment that wraps the Julia simulation.
    Each instance gets a unique simulation ID so that its config,
    policy file, and output directory are unique.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_id, policy_table):
        super(SimulationEnv, self).__init__()
        self.sim_id = sim_id  # e.g., "3_0" for episode 3, simulation 0
        self.policy_table = policy_table
        self.trajectory = []
        self.rewards = []
        self.current_step = 0
        self.done = False
        # Filenames for this simulation
        self.config_file = os.path.join(CONFIG_DIR, f"config_{self.sim_id}.toml")
        self.policy_file = os.path.join(POLICY_DIR, f"policy_{self.sim_id}.json")

    def _update_config(self):
        # Load base config, update OUTPUTDIR, discretization parameters, and policy file path,
        # then save to a unique config file.
        with open(CONFIG_PATH, "r") as f:
            config = toml.load(f)

        config["OUTPUTDIR"] = os.path.join(OUTPUT_DIR, self.sim_id)
        config["S_DISCRETIZATION"] = S_DISCRETIZATION
        config["I_DISCRETIZATION"] = I_DISCRETIZATION
        config["POLICY_FILE"] = self.policy_file
        with open(self.config_file, "w") as f:
            toml.dump(config, f)

    def _run_simulation(self):
        # 1. Update config file and save policy file for this simulation.
        self._update_config()
        save_policy(self.policy_table, filename=self.policy_file)

        # 2. Run the Julia simulation (blocking call).
        try:
            subprocess.run(JULIA_CMD + [self.config_file], check=True)
        except subprocess.CalledProcessError as e:
            logging.error("Error running simulation %s: %s", self.sim_id, e)
            raise

        # 3. Parse the CSV output from the simulation.
        output_dir = os.path.join(OUTPUT_DIR, self.sim_id)
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV output found in " + output_dir)
        csv_file = os.path.join(output_dir, csv_files[0])
        df = pd.read_csv(csv_file)

        # 4. Extract a trajectory of states from weekly summaries.
        #    Also record infection levels for each state.
        weekly_df = df[df["Day"] % 7 == 0].sort_values("Day")
        trajectory = []

        total_adults = None
        total_students = None
        for idx, (i, row) in enumerate(weekly_df.iterrows()):
            if total_adults is None and total_students is None:
                total_students = sum(row[col] for col in row.index if "Students" in col)
                total_adults = sum(row[col] for col in row.index if "Adults" in col)

            S_students = row["Students - Susceptible - Mumbai"] / total_students
            S_adults = row["Adults - Susceptible - Mumbai"] / total_adults

            I_students = row["Students - Infected - Mumbai"] / total_students
            I_adults = row["Adults - Infected - Mumbai"] / total_adults

            state = np.array(
                [S_students, S_adults, I_students, I_adults], dtype=np.float32
            )
            key = state_key(state)
            # action_index = self.policy_table.get(key, 0)
            action_index = self.policy_table.get[key]
            trajectory.append((state, action_index))

        self.trajectory = trajectory  # Save full trajectory

        # 5. Compute per-transition rewards.
        rewards = []
        # For each transition from s_i to s_{i+1} (i from 0 to T-2)
        for i in range(len(trajectory) - 1):
            # Use the action chosen for the transition
            state, action = trajectory[i]
            action = index_to_action(action)

            # Compute intervention cost at this transition.
            cost_t = (action[0] * total_adults + action[1] * total_students) / 7
            immediate_reward = -ALPHA_INTERVENTION * cost_t

            # Infection penalty based on next state's infection fraction.
            I_next = trajectory[i + 1][0][2] + trajectory[i + 1][0][3]
            infection_penalty = -BETA_PEAK * max(0, I_next - I_CRITICAL)

            reward = immediate_reward + infection_penalty
            rewards.append(reward)

        # 6. For the final state, add the terminal outcome penalty.
        final_row = df.iloc[-1]
        total_rec = sum(final_row[col] for col in final_row.index if "Recovered" in col)
        terminal_reward = -LAMBDA_TERMINAL * total_rec
        rewards.append(terminal_reward)

        # Now, len(rewards) equals len(trajectory)
        self.rewards = rewards
        self.total_reward = sum(rewards)

        logging.info(
            "Sim %s: Trajectory length: %d, Total reward: %s, Total recovered: %s",
            self.sim_id,
            len(self.trajectory),
            self.total_reward,
            total_rec,
        )

    def reset(self):
        # Run simulation and return the initial state.
        self._run_simulation()
        return self.trajectory[0][0]


class QNetwork(nn.Module):
    def __init__(
        self, state_dim=4, num_actions=NUM_ACTIONS, hidden_dim=256, num_layers=6
    ):
        super(QNetwork, self).__init__()
        # NEW: Input dimension is state_dim + num_actions (one-hot for action)
        input_dim = state_dim + num_actions
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        # NEW: Output is a single scalar (Q-value)
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        # NEW: Convert action (discrete index) into a one-hot vector and concatenate with state.
        # Expecting state shape: (batch, state_dim), action shape: (batch,) or (batch,1)
        if action.dim() == 2:
            action = action.squeeze(1)
        one_hot_action = F.one_hot(action, num_classes=NUM_ACTIONS).float()
        x = torch.cat([state, one_hot_action], dim=1)
        return self.model(x)


class NNAgent:
    def __init__(self, ckpt=None):
        # ckpt is a dir with 2 files: 'net.pth' and 'target_net.pth'

        self.net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)

        if ckpt:
            self.net.load_state_dict(
                torch.load(os.path.join(ckpt, "net.pth"), map_location=device)
            )
            self.target_net.load_state_dict(
                torch.load(os.path.join(ckpt, "target_net.pth"), map_location=device)
            )

        else:
            # Initialize target network with the same weights as the main network
            self.target_net.load_state_dict(self.net.state_dict())

        self.target_net.eval()  # target network is not directly trained

        # Cache state grid for policy table evaluation (avoids recomputing each time)

        precision_s = len(str(S_DISCRETIZATION).split(".")[-1])
        precision_i = len(str(I_DISCRETIZATION).split(".")[-1])
        s_vals = np.round(
            np.arange(S_min, 1 + S_DISCRETIZATION, S_DISCRETIZATION), precision_s
        )
        i_vals = np.round(
            np.arange(0, I_max + I_DISCRETIZATION, I_DISCRETIZATION), precision_i
        )
        grid = np.array(np.meshgrid(s_vals, s_vals, i_vals, i_vals, indexing="ij"))
        self._states = grid.reshape(4, -1).T  # shape: (num_states, 4)
        self._keys = [f"{row[0]},{row[1]},{row[2]},{row[3]}" for row in self._states]

        self.optimizer = optim.Adam(self.net.parameters(), lr=ALPHA * 0.1)
        # Experience replay buffer
        self.replay_buffer_capacity = 5000  # set a maximum capacity (adjust as needed)
        self.replay_buffer = deque(maxlen=self.replay_buffer_capacity)

        self.batch_size = 24
        self.update_steps = 20  # Number of gradient descent steps per update

        # New: frequency (in gradient steps) to update the target network
        self.target_update_freq = 60
        self.update_counter = 0

    def update(self, trajectory, rewards):
        T = len(trajectory)
        for t in range(T):
            s, a = trajectory[t]
            r = rewards[t]
            if t < T - 1:
                s_next, _ = trajectory[t + 1]
                done = False
            else:
                s_next = trajectory[t][0]
                done = True
            self.replay_buffer.append((s, a, r, s_next, done))

        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.update_steps):
            batch = random.sample(list(self.replay_buffer), self.batch_size)
            states, actions, rewards_batch, next_states, dones = zip(*batch)

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(
                device
            )
            actions_tensor = (
                torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            )
            rewards_tensor = (
                torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1).to(device)
            )
            next_states_tensor = torch.tensor(
                np.array(next_states), dtype=torch.float32
            ).to(device)
            dones_tensor = (
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
            )

            # NEW: Compute Q(s, a) using the state-action network
            q_vals = self.net(states_tensor, actions_tensor)  # shape: (batch, 1)

            # NEW: Compute target Q value using Double DQN style over all possible actions for next state.
            with torch.no_grad():
                b = next_states_tensor.shape[0]
                state_dim = next_states_tensor.shape[1]
                # Expand next states for all candidate actions
                next_states_expanded = next_states_tensor.unsqueeze(1).repeat(
                    1, NUM_ACTIONS, 1
                )  # shape (b, NUM_ACTIONS, state_dim)
                candidate_actions = torch.arange(NUM_ACTIONS).to(device)
                candidate_actions_expanded = candidate_actions.unsqueeze(0).repeat(
                    b, 1
                )  # shape (b, NUM_ACTIONS)
                next_states_flat = next_states_expanded.reshape(-1, state_dim)
                candidate_actions_flat = candidate_actions_expanded.reshape(-1)
                q_next_flat = self.target_net(
                    next_states_flat, candidate_actions_flat
                )  # shape: (b*NUM_ACTIONS, 1)
                q_next = q_next_flat.view(b, NUM_ACTIONS)  # shape: (b, NUM_ACTIONS)
                next_q_values_target, _ = torch.max(q_next, dim=1, keepdim=True)
                target = rewards_tensor + GAMMA * next_q_values_target * (
                    1 - dones_tensor
                )

            loss = nn.MSELoss()(q_vals, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())
                torch.save(self.net.state_dict(), os.path.join(MODEL_DIR, "net.pth"))
                torch.save(
                    self.target_net.state_dict(),
                    os.path.join(MODEL_DIR, "target_net.pth"),
                )

    def get_policy_table(self, eps):
        states_tensor = torch.tensor(self._states, dtype=torch.float32).to(device)
        with torch.no_grad():
            b = states_tensor.shape[0]
            state_dim = states_tensor.shape[1]
            states_expanded = states_tensor.unsqueeze(1).repeat(
                1, NUM_ACTIONS, 1
            )  # shape: (b, NUM_ACTIONS, state_dim)
            candidate_actions = torch.arange(NUM_ACTIONS).to(device)
            candidate_actions_expanded = candidate_actions.unsqueeze(0).repeat(
                b, 1
            )  # shape: (b, NUM_ACTIONS)
            states_flat = states_expanded.reshape(-1, state_dim)
            candidate_actions_flat = candidate_actions_expanded.reshape(-1)
            q_values_flat = self.net(states_flat, candidate_actions_flat)
            q_values = q_values_flat.view(b, NUM_ACTIONS)
            best_actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # NEW: Apply exploration; override best actions with random actions with probability eps.
        rand_mask = np.random.rand(len(self._states)) < eps
        random_actions = np.random.randint(0, NUM_ACTIONS, size=len(self._states))
        final_actions = np.where(rand_mask, random_actions, best_actions)
        policy_table = {
            key: int(action) for key, action in zip(self._keys, final_actions)
        }
        return policy_table


# Function to run a single simulation
###############################


def run_simulation(sim_id, agent: NNAgent, eps):
    """
    Runs a single simulation with a unique sim_id and returns its trajectory and rewards.
    """
    policy_table = agent.get_policy_table(
        eps
    )  # Get the policy table for this simulation
    env = SimulationEnv(sim_id, policy_table)
    env.reset()  # This triggers the simulation run.
    return env.trajectory, env.rewards


###############################
# Main Training Loop (with parallel simulations)
###############################
def plot_loss(losses):
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss vs Episode")
    plt.savefig(LOSS_FILE)
    plt.close()


def train():
    # Initialize agent.
    ckpt = None
    agent = NNAgent(ckpt)
    # Global policy_table is initially empty (defaults to action index 0).
    policy_table = {}
    losses = []

    num_parallel = 1  # Number of parallel simulations per episode

    for episode in range(NUM_EPISODES):
        logging.info("Starting Episode %d", episode)
        trajectories = []
        rewards_list = []
        futures = []

        # Create a thread pool to run simulations in parallel.
        with ThreadPoolExecutor(max_workers=num_parallel) as executor:
            for i in range(num_parallel):
                sim_id = f"{episode}_{i}"
                futures.append(
                    executor.submit(run_simulation, sim_id, agent, EPSILON[episode])
                )
            # Wait for all simulations to complete.
            for future in as_completed(futures):
                traj, rwd = future.result()
                trajectories.append(traj)
                rewards_list.append(rwd)

        # Use each trajectory to update the agent.
        for traj, rwd in zip(trajectories, rewards_list):
            agent.update(traj, rwd)

        # Compute average loss (using absolute terminal reward) over parallel sims.
        avg_loss = np.mean([abs(sum(rwd)) for rwd in rewards_list])
        losses.append(avg_loss)
        logging.info(
            "Episode %d complete. Average Terminal reward: %s", episode, avg_loss
        )

        # (Optionally, save a global policy file for inspection.)
        policy_table = agent.get_policy_table(0)
        save_policy(policy_table, filename=BASE_POLICY_PATH)
        plot_loss(losses)


if __name__ == "__main__":
    train()
