import os
from functools import partial
from operator import truediv
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import Array
import jaxatari.rendering.atraJaxis as aj
import numpy as np

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

# Object sizes and initial positions from Ram State
PLAYER_POSITION = 76, 100
PLAYER_COLOR = (213, 130, 74)
# MAX number of Objects
MAX_PLAYER = 1
MAX_PLAYER_PROJECTILE = 1
MAX_PHOENIX = 8
MAX_BATS = 7
MAX_BOSS = 1
MAX_BOSS_BLOCK_GREEN = 1
MAX_BOSS_BLOCK_BLUE = 48
MAX_BOSS_BLOCK_RED = 104



# === GAME STATE ===
class PhoenixState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array

class PhoenixOberservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array

class PhoenixInfo(NamedTuple):
    step_counter: jnp.ndarray

class CarryState(NamedTuple):
    score: chex.Array

class EntityPosition(NamedTuple):## not sure
    x: chex.Array
    y: chex.Array


def load_sprites(): # load Sprites
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load individual sprite frames
    player_sprites = aj.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player.npy"))
    SPRITE_PLAYER = jnp.expand_dims(player_sprites, axis=0)
    bg_sprites = aj.loadFrame(os.path.join(MODULE_DIR, "./sprites/pong/background.npy"))
    BG_SPRITE = jnp.expand_dims(bg_sprites, axis=0)
    print("Player sprite shape:", SPRITE_PLAYER.shape)
    # shape (5, 10,4)
    return (
        SPRITE_PLAYER,
        BG_SPRITE,

    )
# load sprites on module layer
(SPRITE_PLAYER, SPRITE_BG) = load_sprites()





@jax.jit
def player_step(state: PhoenixState, action: Action) -> Tuple[chex.Array]:
    """Step function for the player."""
    left = jnp.any(
        jnp.array(
            [
                action == Action.LEFT,
                action == Action.UPLEFT,
                action == Action.DOWNLEFT,
                action == Action.LEFTFIRE,
                action == Action.UPLEFTFIRE,
                action == Action.DOWNLEFTFIRE,
            ]
        )
    )
    right = jnp.any(
        jnp.array(
            [
                action == Action.RIGHT,
                action == Action.UPRIGHT,
                action == Action.DOWNRIGHT,
                action == Action.RIGHTFIRE,
                action == Action.UPRIGHTFIRE,
                action == Action.DOWNRIGHTFIRE,
            ]
        )
    )
    player_x = jnp.where(
        right, state.player_x + 1, jnp.where(left, state.player_x - 1, state.player_x)
    )
    return player_x


#ToDo
class JaxPhoenix(JaxEnvironment[PhoenixState, PhoenixOberservation, PhoenixInfo]):
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PhoenixState) -> PhoenixOberservation:
        player = EntityPosition(x=state.player_x, y=state.player_y)
        return PhoenixOberservation(
            player_x = player[0],
            player_y= player[1],
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PhoenixState, all_rewards: jnp.ndarray) -> PhoenixInfo:
        return PhoenixInfo(
            step_counter=0,
        )
    #ToDo _get_info,_get_env_reward,_get_all_rewards,_get_done

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    def __init__(self):
        super().__init__()
        self.step_counter = 0
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        # Add step counter tracking
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> tuple[PhoenixOberservation, PhoenixState]:
        # Reset the state
        return_state = PhoenixState(
            player_x=jnp.array(PLAYER_POSITION[0]),
            player_y=jnp.array(PLAYER_POSITION[1]),
            step_counter=jnp.array(0),
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state

    def step(self,state, action: chex.Array) -> tuple[PhoenixOberservation, PhoenixState, float, bool, PhoenixInfo]:

        #previous_state = state
        #state = state.reset()
        return_state = PhoenixState(player_x=state.player_x, player_y=state.player_y, step_counter=state.step_counter)
        observation = self._get_observation(return_state)
        env_reward = 0.0 #toDO
        done = True #toDo
        info = self._get_info(state, env_reward)
        observation, return_state, env_reward, done, info

from jaxatari.renderers import AtraJaxisRenderer

class PhoenixRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))
        frame_bg = aj.get_sprite_frame(SPRITE_BG, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg)

        frame_player = aj.get_sprite_frame(SPRITE_PLAYER, 0)
        raster = aj.render_at(raster, state.player_x, state.player_y, frame_player)


        return raster



