"""Utilities to run the Cooking Optimization RL agent inside the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from pathlib import Path
import re
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Recipe:
    name: str
    required_ingredients: Tuple[int, ...]
    vegetarian: int
    spice_level: int
    prep_time: int
    nutrition_score: float


class CookingEnvironment:
    """Environment with the same structure used during RL training."""

    ACTIONS = {
        0: "Recommend Recipe A",
        1: "Recommend Recipe B",
        2: "Recommend Recipe C",
        3: "Recommend Recipe D",
        4: "Recommend Recipe E",
        5: "Suggest ingredient substitution",
        6: "Adjust cooking steps for faster preparation",
    }

    INGREDIENT_KEYS = [
        "vegetables",
        "protein",
        "grains",
        "dairy",
        "spices",
        "oil",
        "herbs",
        "legumes",
    ]

    INGREDIENT_TYPE_KEYWORDS = {
        "vegetables": {
            "tomato", "onion", "potato", "carrot", "broccoli", "spinach", "pepper", "cabbage",
            "cauliflower", "zucchini", "eggplant", "lettuce", "mushroom", "peas", "corn",
        },
        "protein": {
            "chicken", "beef", "pork", "fish", "salmon", "tuna", "egg", "turkey", "shrimp", "meat",
        },
        "grains": {
            "rice", "bread", "pasta", "noodle", "oats", "flour", "quinoa", "barley", "wheat", "cereal",
        },
        "dairy": {
            "milk", "cheese", "yogurt", "butter", "cream", "ghee", "paneer",
        },
        "spices": {
            "pepper", "chili", "cumin", "turmeric", "paprika", "coriander", "masala", "garam", "spice",
        },
        "oil": {
            "oil", "olive oil", "vegetable oil", "canola", "sesame oil", "sunflower oil", "coconut oil",
        },
        "herbs": {
            "basil", "oregano", "thyme", "rosemary", "parsley", "cilantro", "mint", "dill",
        },
        "legumes": {
            "lentil", "bean", "chickpea", "peas", "dal", "black beans", "kidney beans",
        },
    }

    COOKING_METHOD_KEYWORDS = {
        "boil": {"boil", "boiled", "simmer", "poach"},
        "steam": {"steam", "steamed"},
        "grill": {"grill", "grilled", "broil", "broiled"},
        "bake": {"bake", "baked", "roast", "roasted"},
        "fry": {"fry", "fried", "saute", "sauteed", "stir fry", "stir-fry"},
    }

    MEAT_KEYWORDS = {
        "chicken", "beef", "pork", "fish", "salmon", "tuna", "turkey", "shrimp", "bacon", "ham",
        "mutton", "lamb", "sausage", "anchovy", "meat",
    }

    SPICY_KEYWORDS = {
        "chili", "chilli", "jalapeno", "pepper", "cayenne", "paprika", "spicy", "masala", "hot sauce",
    }

    HEALTHY_KEYWORDS = {
        "vegetable", "salad", "grilled", "steamed", "baked", "whole", "fresh", "olive oil",
    }

    UNHEALTHY_KEYWORDS = {
        "fried", "sugar", "butter", "cream", "bacon", "sausage", "deep fry", "sweetened",
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.n_ingredients = len(self.INGREDIENT_KEYS)
        self.recipes: List[Recipe] = [
            Recipe("Recipe A", (0, 1, 2, 3), vegetarian=1, spice_level=1, prep_time=25, nutrition_score=0.85),
            Recipe("Recipe B", (2, 4, 5, 6), vegetarian=0, spice_level=2, prep_time=40, nutrition_score=0.72),
            Recipe("Recipe C", (1, 3, 6, 7), vegetarian=1, spice_level=0, prep_time=18, nutrition_score=0.90),
            Recipe("Recipe D", (0, 2, 5, 7), vegetarian=1, spice_level=1, prep_time=30, nutrition_score=0.78),
            Recipe("Recipe E", (0, 1, 4, 6), vegetarian=0, spice_level=2, prep_time=35, nutrition_score=0.68),
        ]

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

    def sample_random_state(self) -> Tuple[int, ...]:
        ingredient_vector = self.rng.choice([0, 1], size=self.n_ingredients, p=[0.3, 0.7]).astype(int)
        vegetarian_pref = int(self.rng.choice([0, 1], p=[0.45, 0.55]))
        spice_pref = int(self.rng.integers(0, 3))
        time_available = int(self.rng.integers(15, 61))
        time_bucket = self._discretize_time(time_available)
        past_recipe_type = int(self.rng.integers(0, len(self.recipes)))

        estimated_nutrition = []
        for recipe in self.recipes:
            availability_ratio = ingredient_vector[list(recipe.required_ingredients)].mean()
            estimate = 0.55 * recipe.nutrition_score + 0.45 * availability_ratio + self.rng.normal(0, 0.04)
            estimate = float(np.clip(estimate, 0.0, 1.0))
            estimated_nutrition.append(self._discretize_nutrition(estimate))

        return tuple(
            ingredient_vector.tolist()
            + [vegetarian_pref, spice_pref, time_bucket, past_recipe_type]
            + estimated_nutrition
        )

    def build_state_from_constraints(
        self,
        available_ingredients: List[str],
        vegetarian_pref: int,
        spice_pref: int,
        time_available: int,
        past_recipe_type: int,
        nutrition_estimates: List[float],
    ) -> Tuple[int, ...]:
        ingredient_vector = [1 if key in available_ingredients else 0 for key in self.INGREDIENT_KEYS]
        time_bucket = self._discretize_time(int(time_available))
        nutrition_bins = [self._discretize_nutrition(float(v)) for v in nutrition_estimates]

        return tuple(
            ingredient_vector
            + [int(vegetarian_pref), int(spice_pref), int(time_bucket), int(past_recipe_type)]
            + nutrition_bins
        )


class QLearningAgent:
    def __init__(self, action_size: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9996
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.rng = np.random.default_rng(123)

    def _ensure_state(self, state: Tuple[int, ...]) -> None:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float64)

    def choose_action(self, state: Tuple[int, ...], greedy_only: bool = False) -> int:
        self._ensure_state(state)
        if (not greedy_only) and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, self.action_size))
        q_values = self.q_table[state]
        max_q = float(np.max(q_values))
        best_actions = np.flatnonzero(np.isclose(q_values, max_q))
        return int(self.rng.choice(best_actions))

    def update(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...]) -> None:
        self._ensure_state(state)
        self._ensure_state(next_state)
        current = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current + self.alpha * (target - current)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def action_values(self, state: Tuple[int, ...]) -> np.ndarray:
        self._ensure_state(state)
        return self.q_table[state].copy()


class CookingOptimizationRL:
    """High-level wrapper for training/loading and app-time recommendation."""

    def __init__(self, seed: int = 42):
        self.env = CookingEnvironment(seed=seed)
        self.agent = QLearningAgent(action_size=len(self.env.ACTIONS))

    def train(self, episodes: int = 6000) -> None:
        for _ in range(episodes):
            state = self.env.sample_random_state()
            action = self.agent.choose_action(state)
            reward, next_state = self._simulate_transition(state, action)
            self.agent.update(state, action, reward, next_state)
            self.agent.decay_epsilon()

    def _simulate_transition(self, state: Tuple[int, ...], action: int) -> Tuple[float, Tuple[int, ...]]:
        ingredient_vector = np.array(state[:8], dtype=int)
        vegetarian_pref = int(state[8])
        spice_pref = int(state[9])
        time_bucket = int(state[10])
        past_recipe_type = int(state[11])
        nutrition_bins = state[12:12 + len(self.env.recipes)]

        time_available = [18, 30, 50][time_bucket]
        is_substitution = action == 5
        is_fast = action == 6

        if 0 <= action < len(self.env.recipes):
            recipe_idx = action
        else:
            recipe_idx = int(np.argmax(nutrition_bins))

        recipe = self.env.recipes[recipe_idx]
        available_count = int(ingredient_vector[list(recipe.required_ingredients)].sum())
        required_count = len(recipe.required_ingredients)
        missing_count = required_count - available_count
        effective_missing = max(0, missing_count - 1) if is_substitution else missing_count
        effective_prep = recipe.prep_time * (0.8 if is_fast else 1.0)

        ingredient_ratio = available_count / required_count
        veg_match = 1.0 if vegetarian_pref == 0 or recipe.vegetarian == 1 else 0.0
        spice_match = 1.0 - abs(recipe.spice_level - spice_pref) / 2.0
        match_score = 0.5 * ingredient_ratio + 0.3 * veg_match + 0.2 * spice_match

        # Encourage variety: re-recommending the same previous profile gets a penalty.
        repeat_penalty = 0.18 if recipe_idx == past_recipe_type else 0.0
        novelty_bonus = 0.06 if recipe_idx != past_recipe_type else 0.0

        time_penalty = max(0.0, (effective_prep - time_available) / 30.0)
        accept_logit = (
            2.2 * match_score
            - 1.2 * time_penalty
            - repeat_penalty
            + novelty_bonus
            + (0.12 if is_substitution else 0.0)
            - 0.25
        )
        accept_prob = 1.0 / (1.0 + np.exp(-accept_logit))
        selected = int(self.agent.rng.random() < accept_prob)

        success_prob = np.clip(0.65 + 0.25 * match_score - 0.18 * (effective_missing / required_count), 0.05, 0.98)
        cooked_success = int(selected and self.agent.rng.random() < success_prob)

        reward = 0.0
        if selected:
            reward += 10
        if cooked_success:
            reward += 8
        if effective_prep <= time_available:
            reward += 5
        if recipe.nutrition_score >= 0.75:
            reward += 4
        if recipe_idx != past_recipe_type:
            reward += 1

        if effective_prep > time_available:
            reward -= 5
        if effective_missing > 0:
            reward -= 6
        if recipe_idx == past_recipe_type:
            reward -= 2
        if selected and (not cooked_success) and available_count > 0:
            reward -= 8
        if not selected:
            reward -= 4

        next_state = self.env.sample_random_state()
        return float(reward), next_state

    def recommend(self, state: Tuple[int, ...]) -> Dict[str, object]:
        q_values, source, avg_distance = self._estimate_q_values(state)

        # If values are flat (common for unseen/weakly-learned states),
        # use a state-aware heuristic to avoid always selecting action 0.
        if np.allclose(q_values, q_values[0]):
            best_action = self._fallback_action_from_state(state)
        else:
            max_q = float(np.max(q_values))
            best_actions = np.flatnonzero(np.isclose(q_values, max_q))
            best_action = int(self.agent.rng.choice(best_actions))

        return {
            "best_action": best_action,
            "best_action_name": self.env.ACTIONS[best_action],
            "q_values": q_values,
            "actions": self.env.ACTIONS,
            "value_source": source,
            "avg_neighbor_distance": avg_distance,
        }

    def _fallback_action_from_state(self, state: Tuple[int, ...]) -> int:
        """Heuristic action for flat Q-values to reduce cold-start tie bias."""
        time_bucket = int(state[10])
        nutrition_bins = list(state[12:12 + len(self.env.recipes)])

        # Prefer fastest strategy when time is tight.
        if time_bucket == 0:
            return 6

        # Otherwise choose recipe profile with strongest estimated nutrition.
        if nutrition_bins:
            return int(np.argmax(nutrition_bins))

        return 0

    def _estimate_q_values(self, state: Tuple[int, ...], top_k: int = 25) -> Tuple[np.ndarray, str, float]:
        """Estimate action values for a state.

        If the exact state exists in the Q-table, return exact values.
        Otherwise, approximate values by averaging nearest states in Hamming distance.
        """
        if state in self.agent.q_table:
            return self.agent.q_table[state].copy(), "exact", 0.0

        if not self.agent.q_table:
            return np.zeros(self.agent.action_size, dtype=np.float64), "empty", float(len(state))

        known_states = list(self.agent.q_table.keys())
        state_arr = np.array(state, dtype=np.int16)

        distances = []
        for known in known_states:
            known_arr = np.array(known, dtype=np.int16)
            # Hamming distance over the discretized state vector.
            dist = int(np.sum(state_arr != known_arr))
            distances.append(dist)

        distances_arr = np.array(distances)
        k = min(top_k, len(known_states))
        nearest_idx = np.argsort(distances_arr)[:k]

        nearest_q_values = np.array([self.agent.q_table[known_states[i]] for i in nearest_idx])
        # Distance-weighted average, giving more influence to closer states.
        weights = 1.0 / (1.0 + distances_arr[nearest_idx])
        weights = weights / weights.sum()
        estimated_q = np.average(nearest_q_values, axis=0, weights=weights)

        return estimated_q, "nearest_neighbors", float(np.mean(distances_arr[nearest_idx]))

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.agent.q_table,
            "epsilon": self.agent.epsilon,
            "model_version": 3,
            "action_size": self.agent.action_size,
            "recipe_count": len(self.env.recipes),
        }
        joblib.dump(payload, model_path)

    def load(self, model_path: Path) -> bool:
        if not model_path.exists():
            return False
        payload = joblib.load(model_path)

        if int(payload.get("model_version", -1)) != 3:
            return False

        if int(payload.get("action_size", -1)) != self.agent.action_size:
            return False
        if int(payload.get("recipe_count", -1)) != len(self.env.recipes):
            return False

        loaded_q_table = payload.get("q_table", {})
        if loaded_q_table:
            sample_q = next(iter(loaded_q_table.values()))
            if len(sample_q) != self.agent.action_size:
                return False

        self.agent.q_table = loaded_q_table
        self.agent.epsilon = float(payload.get("epsilon", 0.02))
        return True

    @staticmethod
    def _compact_text(raw_value: object) -> str:
        """Convert a potentially list-like string column into searchable lowercase text."""
        if raw_value is None:
            return ""
        text = str(raw_value).lower()
        return (
            text.replace("[", " ")
            .replace("]", " ")
            .replace('"', " ")
            .replace("'", " ")
            .replace(",", " ")
        )

    @staticmethod
    def _estimate_time_minutes(directions_text: str) -> float:
        """Estimate total recipe time by summing explicit minute/hour mentions in directions."""
        if not directions_text:
            return 35.0

        minute_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:minute|min)", directions_text)
        hour_hits = re.findall(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)", directions_text)

        minutes = sum(float(x) for x in minute_hits)
        minutes += 60.0 * sum(float(x) for x in hour_hits)

        if minutes <= 0:
            return 35.0
        return float(min(minutes, 360.0))

    def _estimate_spice_level(self, combined_text: str) -> int:
        spicy_hits = sum(1 for kw in self.env.SPICY_KEYWORDS if kw in combined_text)
        if spicy_hits >= 3:
            return 2
        if spicy_hits >= 1:
            return 1
        return 0

    def _nutrition_proxy(self, combined_text: str) -> float:
        healthy_hits = sum(1 for kw in self.env.HEALTHY_KEYWORDS if kw in combined_text)
        unhealthy_hits = sum(1 for kw in self.env.UNHEALTHY_KEYWORDS if kw in combined_text)
        raw = 0.5 + 0.08 * healthy_hits - 0.08 * unhealthy_hits
        return float(np.clip(raw, 0.0, 1.0))

    def _strategy_target_recipe_idx(self, action: int, nutrition_estimates: List[float]) -> int:
        if 0 <= action < len(self.env.recipes):
            return action
        return int(np.argmax(nutrition_estimates))

    def rank_dataset_recipes(
        self,
        dataset_path: Path,
        available_ingredients: List[str],
        available_ingredient_items: List[str] | None,
        available_cooking_methods: List[str] | None,
        vegetarian_pref: int,
        spice_pref: int,
        time_available: int,
        best_action: int,
        nutrition_estimates: List[float],
        top_n: int = 5,
        chunk_size: int = 50000,
        max_rows: int | None = None,
    ) -> Dict[str, object]:
        """Rank recipes from the full dataset using RL strategy + user constraints.

        Uses fully vectorized pandas/numpy operations for speed on 2M+ row datasets.
        Processes the CSV in chunks to keep memory usage low.

        Args:
            max_rows: When set, only the first ``max_rows`` rows are read (quick-scan
                      mode).  Set to ``None`` to scan the entire dataset.
        """
        if not dataset_path.exists():
            return {"recipes": [], "scanned_rows": 0, "warning": f"Dataset not found: {dataset_path}"}

        selected_groups = [g for g in available_ingredients if g in self.env.INGREDIENT_TYPE_KEYWORDS]
        selected_items = [i.strip().lower() for i in (available_ingredient_items or []) if str(i).strip()]
        selected_methods = [
            m for m in (available_cooking_methods or []) if m in self.env.COOKING_METHOD_KEYWORDS
        ]

        # Build regex pattern strings once, outside the chunk loop.
        group_pat_strs: Dict[str, str] = {
            g: '|'.join(re.escape(kw) for kw in self.env.INGREDIENT_TYPE_KEYWORDS[g])
            for g in selected_groups
        }
        method_pat_strs: Dict[str, str] = {
            m: '|'.join(re.escape(kw) for kw in self.env.COOKING_METHOD_KEYWORDS[m])
            for m in selected_methods
        }
        meat_pat = '|'.join(re.escape(kw) for kw in self.env.MEAT_KEYWORDS)
        healthy_pat = '|'.join(re.escape(kw) for kw in self.env.HEALTHY_KEYWORDS)
        unhealthy_pat = '|'.join(re.escape(kw) for kw in self.env.UNHEALTHY_KEYWORDS)
        spicy_pats = [re.escape(kw) for kw in self.env.SPICY_KEYWORDS]

        target_recipe_idx = self._strategy_target_recipe_idx(best_action, nutrition_estimates)
        target_nutrition = self.env.recipes[target_recipe_idx].nutrition_score

        all_candidates: List[Dict[str, object]] = []
        scanned_rows = 0

        csv_kwargs: dict = dict(
            usecols=["title", "ingredients", "directions", "link", "source", "NER"],
            dtype=str,
            low_memory=True,
        )
        if max_rows is not None:
            csv_kwargs["nrows"] = int(max_rows)
        csv_kwargs["chunksize"] = chunk_size

        for chunk in pd.read_csv(dataset_path, **csv_kwargs):
            scanned_rows += len(chunk)

            # Fill NaN and reset index for clean positional access.
            chunk = chunk.fillna("").reset_index(drop=True)

            titles_s = chunk["title"].str.strip()
            valid = titles_s.ne("").values
            if not valid.any():
                continue

            c = chunk[valid].reset_index(drop=True)

            # Keep only rows that contain usable recipe text fields.
            ingredients_s = c["ingredients"].fillna("").astype(str).str.strip()
            ner_s = c["NER"].fillna("").astype(str).str.strip()
            directions_s = c["directions"].fillna("").astype(str).str.strip()

            invalid_tokens = {"", "nan", "none", "[]", "{}"}
            ingredients_ok = ~ingredients_s.str.lower().isin(invalid_tokens)
            ner_ok = ~ner_s.str.lower().isin(invalid_tokens)
            directions_ok = ~directions_s.str.lower().isin(invalid_tokens)

            usable_mask = ingredients_ok | ner_ok | directions_ok
            if not usable_mask.any():
                continue

            c = c[usable_mask].reset_index(drop=True)
            titles_arr = c["title"].str.strip().values

            # Build combined text series (title + ingredients + NER, all lowercase).
            combined = (
                c["title"].str.lower() + " "
                + c["ingredients"].str.lower() + " "
                + c["NER"].str.lower()
            )
            combined_with_dir = combined + " " + c["directions"].str.lower()

            # ---- Ingredient availability score --------------------------------
            # Fraction of selected ingredient group keywords detected in recipe.
            if group_pat_strs:
                group_hits = np.stack(
                    [
                        combined.str.contains(pat, regex=True, na=False).astype(np.float32).values
                        for pat in group_pat_strs.values()
                    ],
                    axis=1,
                )
                availability_scores: np.ndarray = group_hits.mean(axis=1)
            else:
                availability_scores = np.full(len(c), 0.5, dtype=np.float32)

            # ---- Explicit ingredient-item score -------------------------------
            # Uses user-entered ingredient items (e.g., "chicken") for stricter matching.
            if selected_items:
                item_hits = np.stack(
                    [
                        combined.str.contains(re.escape(item), regex=True, na=False).astype(np.float32).values
                        for item in selected_items
                    ],
                    axis=1,
                )
                ingredient_item_scores: np.ndarray = item_hits.mean(axis=1)
                item_presence_mask = item_hits.sum(axis=1) > 0
                if not item_presence_mask.any():
                    continue

                c = c[item_presence_mask].reset_index(drop=True)
                titles_arr = c["title"].str.strip().values
                combined = (
                    c["title"].str.lower() + " "
                    + c["ingredients"].str.lower() + " "
                    + c["NER"].str.lower()
                )
                combined_with_dir = combined + " " + c["directions"].str.lower()
                availability_scores = availability_scores[item_presence_mask]
                ingredient_item_scores = ingredient_item_scores[item_presence_mask]
            else:
                ingredient_item_scores = np.full(len(c), 0.5, dtype=np.float32)

            # ---- Vegetarian score --------------------------------------------
            is_non_veg: np.ndarray = combined.str.contains(meat_pat, regex=True, na=False).values
            veg_scores = np.where(vegetarian_pref == 0, 1.0, (~is_non_veg).astype(np.float32))

            # ---- Spice level score -------------------------------------------
            # Count number of distinct spicy keywords found per recipe.
            spice_counts = np.zeros(len(c), dtype=np.int16)
            for sp in spicy_pats:
                spice_counts += combined_with_dir.str.contains(sp, regex=True, na=False).astype(np.int16).values
            recipe_spice = np.where(spice_counts >= 3, 2, np.where(spice_counts >= 1, 1, 0))
            spice_scores = 1.0 - np.abs(recipe_spice.astype(np.float32) - float(spice_pref)) / 2.0

            # ---- Time estimation --------------------------------------------
            # Extract first minute/hour mention from directions as prep time proxy.
            dirs_lower = c["directions"].str.lower()
            mins_found = (
                dirs_lower.str.extract(r"(\d+)\s*(?:minute|min)", expand=False)
                .fillna("0").astype(float).values
            )
            hrs_found = (
                dirs_lower.str.extract(r"(\d+)\s*(?:hour|hr)", expand=False)
                .fillna("0").astype(float).values
            )
            est_times = np.clip(mins_found + 60.0 * hrs_found, 0.0, 360.0)
            est_times = np.where(est_times <= 0, 35.0, est_times)
            time_fits = np.maximum(
                0.0,
                1.0 - np.abs(est_times - float(time_available)) / max(20.0, float(time_available)),
            )
            time_bonuses = (est_times <= float(time_available)).astype(np.float32) * 0.1

            # ---- Cooking method score --------------------------------------
            if method_pat_strs:
                method_hits = np.stack(
                    [
                        dirs_lower.str.contains(pat, regex=True, na=False).astype(np.float32).values
                        for pat in method_pat_strs.values()
                    ],
                    axis=1,
                )
                method_scores: np.ndarray = method_hits.mean(axis=1)
            else:
                method_scores = np.full(len(c), 0.5, dtype=np.float32)

            # ---- Nutrition proxy --------------------------------------------
            # Binary presence of healthy/unhealthy keywords → continuous score.
            h_hits = combined_with_dir.str.contains(healthy_pat, regex=True, na=False).astype(np.float32).values
            u_hits = combined_with_dir.str.contains(unhealthy_pat, regex=True, na=False).astype(np.float32).values
            nutrition_proxies = np.clip(0.5 + 0.08 * h_hits - 0.08 * u_hits, 0.0, 1.0)
            nutrition_fits = 1.0 - np.abs(nutrition_proxies - target_nutrition)

            # ---- Strategy bonus ---------------------------------------------
            strategy_bonus = np.zeros(len(c), dtype=np.float32)
            if best_action == 5:
                # Substitution strategy: favor recipes where some ingredients missing.
                strategy_bonus += (availability_scores < 0.35).astype(np.float32) * 0.08
            if best_action == 6:
                # Fast-prep strategy: favor recipes close to (but not over) time limit.
                strategy_bonus += (est_times <= float(time_available)).astype(np.float32) * 0.08

            # ---- Final weighted score ---------------------------------------
            scores = (
                0.30 * availability_scores
                + 0.18 * ingredient_item_scores
                + 0.15 * veg_scores
                + 0.14 * spice_scores
                + 0.14 * time_fits
                + 0.14 * nutrition_fits
                + 0.05 * method_scores
                + time_bonuses
                + strategy_bonus
            ).astype(np.float32)

            # Keep only top candidates per chunk to limit memory.
            buffer = min(top_n * 4, len(scores))
            top_local = np.argpartition(scores, -buffer)[-buffer:]

            link_arr = c["link"].values
            source_arr = c["source"].values
            ingredients_arr = c["ingredients"].values
            ner_arr = c["NER"].values
            directions_arr = c["directions"].values
            for idx in top_local:
                ingredients_raw = str(ingredients_arr[idx]).strip()
                ner_raw = str(ner_arr[idx]).strip()
                directions_raw = str(directions_arr[idx]).strip()

                ingredients_text = ingredients_raw if ingredients_raw.lower() not in invalid_tokens else ""
                if not ingredients_text and ner_raw.lower() not in invalid_tokens:
                    ingredients_text = ner_raw

                directions_text = directions_raw if directions_raw.lower() not in invalid_tokens else ""
                all_candidates.append(
                    {
                        "title": str(titles_arr[idx]),
                        "source": str(source_arr[idx]),
                        "link": str(link_arr[idx]),
                        "ingredients_text": ingredients_text,
                        "directions_text": directions_text,
                        "score": float(scores[idx]),
                        "estimated_time_min": float(est_times[idx]),
                        "vegetarian_compatible": bool(not is_non_veg[idx]),
                    }
                )

        all_candidates.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type]
        return {
            "recipes": all_candidates[:top_n],
            "scanned_rows": scanned_rows,
            "warning": "",
        }
