from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jaxatari.environment import JaxEnvironment
import chex


# === GAME STATE ===
class PhoenixState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_bullet_y: chex.Array
    bullet_active: chex.Array
    enemy_positions: chex.Array  # shape (N, 2)
    enemy_alive: chex.Array  # shape (N,)
    score: chex.Array
    step_count: chex.Array


# === PURE STEP FUNCTION ===
@jax.jit
def phoenix_step(state: PhoenixState, action: chex.Array, rng: jax.random.PRNGKey) -> Tuple[chex.Array, PhoenixState, chex.Array, chex.Array, dict]:
    # Unpack state
    player_x = state.player_x
    player_y = state.player_y
    bullet_y = state.player_bullet_y
    bullet_active = state.bullet_active
    enemy_positions = state.enemy_positions
    enemy_alive = state.enemy_alive
    score = state.score
    step_count = state.step_count

    # === Handle player movement ===
    move = jnp.where(action == 1, -1, jnp.where(action == 2, 1, 0))
    new_player_x = jnp.clip(player_x + move, 0, 19)

    # === Handle shooting ===
    shoot = jnp.where(action == 3, 1, 0)
    new_bullet_active = jnp.where((bullet_active == 0) & (shoot == 1), 1, bullet_active)
    new_bullet_y = jnp.where(new_bullet_active == 1, bullet_y - 1, bullet_y)

    # === Check for hits ===
    hit_mask = (new_bullet_active == 1) & jnp.any(
        (enemy_alive[:, None]) &
        (enemy_positions[:, 0] == new_player_x) &
        (enemy_positions[:, 1] == new_bullet_y),
        axis=0
    )
    updated_enemy_alive = jnp.where(
        (enemy_positions[:, 0] == new_player_x) & (enemy_positions[:, 1] == new_bullet_y),
        0,
        enemy_alive
    )
    updated_score = score + jnp.sum(enemy_alive - updated_enemy_alive)

    # === Bullet reset if it goes off screen or hits ===
    new_bullet_active = jnp.where((new_bullet_y < 0) | hit_mask, 0, new_bullet_active)
    new_bullet_y = jnp.where((new_bullet_y < 0) | hit_mask, player_y - 1, new_bullet_y)

    # === Game over if all enemies are dead ===
    done = jnp.all(updated_enemy_alive == 0)

    # === Observation: for now, just use player x/y and score ===
    obs = jnp.array([new_player_x, new_player_y, updated_score], dtype=jnp.int32)

    new_state = PhoenixState(
        player_x=new_player_x,
        player_y=player_y,
        player_bullet_y=new_bullet_y,
        bullet_active=new_bullet_active,
        enemy_positions=enemy_positions,
        enemy_alive=updated_enemy_alive,
        score=updated_score,
        step_count=step_count + 1,
    )

    return obs, new_state, updated_score, done, {}


# === ENVIRONMENT CLASS ===
class JaxPhoenix(JaxEnvironment):
    def __init__(self):
        self.num_enemies = 5

    def reset(self, rng: jax.random.PRNGKey) -> Tuple[chex.Array, PhoenixState]:
        enemy_xs = jnp.linspace(0, 19, self.num_enemies, dtype=jnp.int32)
        enemy_positions = jnp.stack([enemy_xs, jnp.ones_like(enemy_xs) * 2], axis=-1)

        state = PhoenixState(
            player_x=jnp.array(10, dtype=jnp.int32),
            player_y=jnp.array(0, dtype=jnp.int32),
            player_bullet_y=jnp.array(-1, dtype=jnp.int32),
            bullet_active=jnp.array(0, dtype=jnp.int32),
            enemy_positions=enemy_positions,
            enemy_alive=jnp.ones(self.num_enemies, dtype=jnp.int32),
            score=jnp.array(0, dtype=jnp.int32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        obs = jnp.array([state.player_x, state.player_y, state.score], dtype=jnp.int32)
        return obs, state

    def step(self, state: PhoenixState, action: chex.Array) -> Tuple[chex.Array, PhoenixState, chex.Array, chex.Array, dict]:
        rng = jax.random.PRNGKey(0)  # Replace with proper RNG handling
        return phoenix_step(state, action, rng)

    def get_action_space(self) -> chex.Array:
        # 0: NOOP, 1: LEFT, 2: RIGHT, 3: FIRE
        return jnp.array([0, 1, 2, 3], dtype=jnp.int32)

