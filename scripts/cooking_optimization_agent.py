"""Cooking Optimization Agent using tabular Q-learning.

This module provides:
1. A custom RL environment for cooking recommendations.
2. A Q-learning agent with epsilon-greedy exploration.
3. Training and evaluation utilities.
4. Visualization of learning progress.

Run:
    python scripts/cooking_optimization_agent.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Recipe:
    """Recipe metadata used by the environment."""

    name: str
    required_ingredients: Tuple[int, ...]
    vegetarian: int
    spice_level: int
    prep_time: int
    nutrition_score: float
    cost: float


class CookingEnvironment:
    """Custom environment for recipe recommendation optimization."""

    ACTIONS = {
        0: "Recommend Recipe A",
        1: "Recommend Recipe B",
        2: "Recommend Recipe C",
        3: "Suggest ingredient substitution",
        4: "Adjust cooking steps for faster preparation",
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_ingredients = 8

        # Recipe A, B, C for discrete recommendation actions.
        self.recipes: List[Recipe] = [
            Recipe(
                name="Recipe A",
                required_ingredients=(0, 1, 2, 3),
                vegetarian=1,
                spice_level=1,
                prep_time=25,
                nutrition_score=0.85,
                cost=6.0,
            ),
            Recipe(
                name="Recipe B",
                required_ingredients=(2, 4, 5, 6),
                vegetarian=0,
                spice_level=2,
                prep_time=40,
                nutrition_score=0.72,
                cost=8.5,
            ),
            Recipe(
                name="Recipe C",
                required_ingredients=(1, 3, 6, 7),
                vegetarian=1,
                spice_level=0,
                prep_time=18,
                nutrition_score=0.9,
                cost=5.0,
            ),
        ]

        self.current_state: Tuple[int, ...] | None = None
        self.current_context: Dict[str, np.ndarray | int | float] = {}

    def _discretize_time(self, minutes: int) -> int:
        if minutes <= 20:
            return 0
        if minutes <= 35:
            return 1
        return 2

    def _discretize_nutrition(self, score: float) -> int:
        if score < 0.45:
            return 0
        if score < 0.75:
            return 1
        return 2

    def _sample_state(self) -> Tuple[int, ...]:
        ingredient_vector = self.rng.choice([0, 1], size=self.n_ingredients, p=[0.3, 0.7]).astype(int)
        vegetarian_pref = int(self.rng.choice([0, 1], p=[0.45, 0.55]))
        spice_pref = int(self.rng.integers(0, 3))
        time_available = int(self.rng.integers(15, 61))
        time_bucket = self._discretize_time(time_available)
        past_recipe_type = int(self.rng.integers(0, 3))

        estimated_nutrition_raw = []
        for recipe in self.recipes:
            availability_ratio = ingredient_vector[list(recipe.required_ingredients)].mean()
            estimate = 0.55 * recipe.nutrition_score + 0.45 * availability_ratio + self.rng.normal(0, 0.04)
            estimate = float(np.clip(estimate, 0.0, 1.0))
            estimated_nutrition_raw.append(estimate)

        estimated_nutrition_bins = tuple(self._discretize_nutrition(v) for v in estimated_nutrition_raw)

        # State vector structure:
        # [ingredient(8), vegetarian_pref, spice_pref, time_bucket,
        #  past_recipe_type, estimated_nutrition_A/B/C]
        state = tuple(
            ingredient_vector.tolist()
            + [vegetarian_pref, spice_pref, time_bucket, past_recipe_type]
            + list(estimated_nutrition_bins)
        )

        self.current_context = {
            "ingredient_vector": ingredient_vector,
            "vegetarian_pref": vegetarian_pref,
            "spice_pref": spice_pref,
            "time_available": time_available,
            "past_recipe_type": past_recipe_type,
            "estimated_nutrition_raw": np.array(estimated_nutrition_raw),
        }

        return state

    def reset(self) -> Tuple[int, ...]:
        self.current_state = self._sample_state()
        return self.current_state

    def _recipe_match_score(self, recipe: Recipe, ingredient_vector: np.ndarray, vegetarian_pref: int, spice_pref: int) -> float:
        ingredient_ratio = float(ingredient_vector[list(recipe.required_ingredients)].mean())
        vegetarian_match = 1.0 if vegetarian_pref == 0 or recipe.vegetarian == 1 else 0.0
        spice_match = 1.0 - (abs(recipe.spice_level - spice_pref) / 2.0)
        return 0.5 * ingredient_ratio + 0.3 * vegetarian_match + 0.2 * spice_match

    def _acceptance_probability(
        self,
        recipe: Recipe,
        match_score: float,
        time_available: int,
        is_substitution_action: bool,
        is_fast_action: bool,
    ) -> float:
        effective_time = recipe.prep_time * (0.8 if is_fast_action else 1.0)
        time_penalty = max(0.0, (effective_time - time_available) / 30.0)
        substitution_bonus = 0.12 if is_substitution_action else 0.0
        z = 2.3 * match_score - 1.3 * time_penalty + substitution_bonus - 0.25
        probability = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(probability, 0.05, 0.98))

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, float | int | str]]:
        if self.current_state is None:
            raise RuntimeError("Call reset() before step().")

        ingredient_vector = self.current_context["ingredient_vector"]
        vegetarian_pref = int(self.current_context["vegetarian_pref"])
        spice_pref = int(self.current_context["spice_pref"])
        time_available = int(self.current_context["time_available"])
        estimated_nutrition_raw = self.current_context["estimated_nutrition_raw"]

        is_substitution_action = action == 3
        is_fast_action = action == 4

        if action in [0, 1, 2]:
            chosen_recipe_idx = action
        else:
            # For strategy actions, choose the best estimated recipe and apply the strategy.
            chosen_recipe_idx = int(np.argmax(estimated_nutrition_raw))

        recipe = self.recipes[chosen_recipe_idx]
        available_count = int(ingredient_vector[list(recipe.required_ingredients)].sum())
        required_count = len(recipe.required_ingredients)
        missing_count = required_count - available_count

        # Substitution can reduce one missing ingredient.
        effective_missing_count = max(0, missing_count - 1) if is_substitution_action else missing_count

        effective_prep_time = recipe.prep_time * (0.8 if is_fast_action else 1.0)
        match_score = self._recipe_match_score(recipe, ingredient_vector, vegetarian_pref, spice_pref)
        accept_prob = self._acceptance_probability(
            recipe, match_score, time_available, is_substitution_action, is_fast_action
        )
        selected = int(self.rng.random() < accept_prob)

        cook_success_prob = 0.65 + 0.25 * match_score - 0.18 * (effective_missing_count / required_count)
        cook_success_prob = float(np.clip(cook_success_prob, 0.05, 0.98))
        cooked_successfully = int(selected and (self.rng.random() < cook_success_prob))

        reward = 0.0

        # Positive rewards
        if selected:
            reward += 10
        if cooked_successfully:
            reward += 8
        if effective_prep_time <= time_available:
            reward += 5
        if recipe.nutrition_score >= 0.75:
            reward += 4

        # Negative rewards
        if effective_prep_time > time_available:
            reward -= 5
        if effective_missing_count > 0:
            reward -= 6
        ingredient_waste = int(selected and (not cooked_successfully) and available_count > 0)
        if ingredient_waste:
            reward -= 8
        if not selected:
            reward -= 4

        info = {
            "action_name": self.ACTIONS[action],
            "recipe_name": recipe.name,
            "selected": selected,
            "cooked_successfully": cooked_successfully,
            "effective_prep_time": float(effective_prep_time),
            "time_available": time_available,
            "time_within_limit": int(effective_prep_time <= time_available),
            "ingredient_missing": effective_missing_count,
            "ingredient_waste": ingredient_waste,
            "nutrition_score": recipe.nutrition_score,
            "acceptance_probability": accept_prob,
        }

        # One-step episodic task. Provide next sampled state for bootstrapped update.
        next_state = self._sample_state()
        self.current_state = next_state
        done = True
        return next_state, float(reward), done, info


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        action_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.9996,
    ):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.state_visits: Dict[Tuple[int, ...], int] = {}
        self.rng = np.random.default_rng(123)

    def _ensure_state(self, state: Tuple[int, ...]) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float64)
            self.state_visits[state] = 0

    def choose_action(self, state: Tuple[int, ...], greedy_only: bool = False) -> int:
        self._ensure_state(state)
        self.state_visits[state] += 1
        if (not greedy_only) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.action_size))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])

        # Q-learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def expected_rewards_for_state(self, state: Tuple[int, ...]) -> np.ndarray:
        self._ensure_state(state)
        return self.q_table[state].copy()


class TrainingManager:
    """Coordinates training, evaluation, and reporting."""

    def __init__(self, env: CookingEnvironment, agent: QLearningAgent):
        self.env = env
        self.agent = agent
        self.training_rewards: List[float] = []
        self.training_success: List[int] = []
        self.training_time_within_limit: List[int] = []
        self.training_waste_events: List[int] = []
        self.training_time_saved: List[float] = []

    def train(self, episodes: int = 5000) -> None:
        for _ in range(episodes):
            state = self.env.reset()
            action = self.agent.choose_action(state)
            next_state, reward, _, info = self.env.step(action)
            self.agent.update(state, action, reward, next_state)
            self.agent.decay_exploration()

            self.training_rewards.append(reward)
            self.training_success.append(int(info["selected"]))
            self.training_time_within_limit.append(int(info["time_within_limit"]))
            self.training_waste_events.append(int(info["ingredient_waste"]))

            # Baseline for time reduction: the same recipe without action-specific speedup.
            raw_recipe_name = str(info["recipe_name"])
            recipe_idx = [r.name for r in self.env.recipes].index(raw_recipe_name)
            baseline_time = self.env.recipes[recipe_idx].prep_time
            time_saved = max(0.0, baseline_time - float(info["effective_prep_time"]))
            self.training_time_saved.append(time_saved)

    def evaluate(self, episodes: int = 1000) -> Dict[str, float]:
        rewards = []
        success_flags = []
        time_within_flags = []
        waste_flags = []
        time_saved_values = []

        for _ in range(episodes):
            state = self.env.reset()
            action = self.agent.choose_action(state, greedy_only=True)
            _, reward, _, info = self.env.step(action)

            rewards.append(reward)
            success_flags.append(int(info["selected"]))
            time_within_flags.append(int(info["time_within_limit"]))
            waste_flags.append(int(info["ingredient_waste"]))

            recipe_idx = [r.name for r in self.env.recipes].index(str(info["recipe_name"]))
            baseline_time = self.env.recipes[recipe_idx].prep_time
            time_saved_values.append(max(0.0, baseline_time - float(info["effective_prep_time"])))

        return {
            "avg_reward": float(np.mean(rewards)),
            "recommendation_success_rate": float(np.mean(success_flags)),
            "time_within_limit_rate": float(np.mean(time_within_flags)),
            "avg_time_reduction_minutes": float(np.mean(time_saved_values)),
            "efficient_ingredient_usage_rate": float(1.0 - np.mean(waste_flags)),
        }

    def policy_table(self, top_n: int = 25) -> pd.DataFrame:
        rows = []
        for state, q_values in self.agent.q_table.items():
            best_action = int(np.argmax(q_values))
            rows.append(
                {
                    "state": state,
                    "visits": self.agent.state_visits.get(state, 0),
                    "best_action": best_action,
                    "best_action_name": self.env.ACTIONS[best_action],
                    "expected_reward_best_action": float(np.max(q_values)),
                    "q_values": q_values.round(3).tolist(),
                }
            )

        policy_df = pd.DataFrame(rows)
        if policy_df.empty:
            return policy_df
        policy_df = policy_df.sort_values(["visits", "expected_reward_best_action"], ascending=False)
        return policy_df.head(top_n).reset_index(drop=True)

    def save_training_plot(self, out_path: Path, moving_window: int = 100) -> None:
        rewards = np.array(self.training_rewards, dtype=float)
        if len(rewards) == 0:
            raise RuntimeError("No rewards found. Train the agent before plotting.")

        moving_avg = pd.Series(rewards).rolling(window=moving_window, min_periods=1).mean().to_numpy()

        plt.figure(figsize=(11, 5))
        plt.plot(rewards, alpha=0.25, linewidth=1.0, label="Episode reward")
        plt.plot(moving_avg, linewidth=2.0, label=f"Moving average ({moving_window})")
        plt.title("Cooking Optimization Agent: Training Reward vs Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=160)
        plt.close()


def describe_state(state: Tuple[int, ...]) -> pd.DataFrame:
    """Return a readable state breakdown for demonstration."""

    ingredient_vector = list(state[:8])
    vegetarian_pref, spice_pref, time_bucket, past_recipe = state[8:12]
    nutrition_bins = state[12:15]

    state_description = {
        "ingredient_vector": [ingredient_vector],
        "vegetarian_pref(0=non-veg,1=veg)": [vegetarian_pref],
        "spice_pref(0=mild,1=medium,2=hot)": [spice_pref],
        "time_bucket(0<=20,1<=35,2>35)": [time_bucket],
        "past_recipe_type(0/1/2)": [past_recipe],
        "nutrition_bins_A_B_C(0/1/2)": [list(nutrition_bins)],
    }
    return pd.DataFrame(state_description)


def main() -> None:
    env = CookingEnvironment(seed=42)
    agent = QLearningAgent(
        action_size=5,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
    )
    manager = TrainingManager(env=env, agent=agent)

    train_episodes = 6000
    manager.train(episodes=train_episodes)
    metrics = manager.evaluate(episodes=1200)

    plot_path = Path("models") / "cooking_optimization" / "training_reward_vs_episode.png"
    manager.save_training_plot(plot_path)

    # Demonstrate output requirements for one learned query state.
    sample_state = max(agent.state_visits.items(), key=lambda kv: kv[1])[0]
    q_values = agent.expected_rewards_for_state(sample_state)
    recommended_action = int(np.argmax(q_values))

    print("\n=== Cooking Optimization Agent (Q-Learning) ===")
    print(f"Training episodes: {train_episodes}")
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")

    print("\nSample State Description:")
    print(describe_state(sample_state).to_string(index=False))

    print("\nExpected Reward for Each Action:")
    for action_id, action_name in env.ACTIONS.items():
        print(f"- Action {action_id} ({action_name}): {q_values[action_id]:.4f}")

    print("\nRecommended Action:")
    print(f"- Action {recommended_action}: {env.ACTIONS[recommended_action]}")

    policy_df = manager.policy_table(top_n=20)
    policy_path = Path("models") / "cooking_optimization" / "learned_policy_top_states.csv"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_df.to_csv(policy_path, index=False)

    print("\nPolicy Table (Top 20 frequently visited states):")
    print(policy_df[["visits", "best_action", "best_action_name", "expected_reward_best_action"]].to_string(index=False))

    print("\nSaved Artifacts:")
    print(f"- Training graph: {plot_path}")
    print(f"- Learned policy table: {policy_path}")


if __name__ == "__main__":
    main()
