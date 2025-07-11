import os
import time
from distutils.dep_util import newer
from functools import partial
from operator import truediv
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame
from gymnasium.envs.tabular.blackjack import score
from jax import Array
from scipy.stats import false_discovery_control
import jaxatari.spaces as spaces

import jaxatari.rendering.jax_rendering_utils as jr
import numpy as np
from enum import Enum
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.spaces import Space

# Phoenix Game by: Florian Schmidt, Finn Keller

# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

# Object sizes and initial positions from Ram State
PLAYER_POSITION = 76, 175
PLAYER_COLOR = (213, 130, 74)
PLAYER_BOUNDS = (0, 155) # (left, right)
# Enemy Positions for level 1(phoenix)
ENEMY_POSITIONS_X = jnp.array([123 - WIDTH//2, 123 -WIDTH//2, 136-WIDTH//2, 136-WIDTH//2, 160-WIDTH//2, 160-WIDTH//2, 174-WIDTH//2, 174-WIDTH//2])
ENEMY_POSITIONS_Y = jnp.array([HEIGHT-135,HEIGHT- 153,HEIGHT- 117,HEIGHT- 171,HEIGHT- 117,HEIGHT- 171,HEIGHT- 135,HEIGHT- 153])

# Enemy Positions for each level
ENEMY_POSITIONS_X_LIST = [
lambda:jnp.array([123 - WIDTH//2, 123 -WIDTH//2, 136-WIDTH//2, 136-WIDTH//2, 160-WIDTH//2, 160-WIDTH//2, 174-WIDTH//2, 174-WIDTH//2]).astype(jnp.int32),
lambda:jnp.array([141 - WIDTH//2, 155 - WIDTH//2, 127- WIDTH//2, 169 - WIDTH//2,134 - WIDTH//2, 162 - WIDTH//2, 120 - WIDTH//2, 176 - WIDTH//2]).astype(jnp.int32),
lambda:jnp.array([123 - WIDTH//2, 170 -WIDTH//2, 123-WIDTH//2, 180-WIDTH//2, 123-WIDTH//2, 170-WIDTH//2,123-WIDTH//2,-1 ]).astype(jnp.int32),
lambda:jnp.array([123 - WIDTH//2, 180 - WIDTH//2, 123- WIDTH//2, 170 - WIDTH//2,123 - WIDTH//2, 180 - WIDTH//2, 123 - WIDTH//2,-1]).astype(jnp.int32),
lambda:jnp.array([150 - WIDTH//2, -1, -1, -1, -1, -1 ,-1 ,-1]).astype(jnp.int32),
]
ENEMY_POSITIONS_Y_LIST = [
lambda:jnp.array([HEIGHT-135,HEIGHT- 153,HEIGHT- 117,HEIGHT- 171,HEIGHT- 117,HEIGHT- 171,HEIGHT- 135,HEIGHT- 153]).astype(jnp.int32),
lambda:jnp.array([HEIGHT-171, HEIGHT-171, HEIGHT-135, HEIGHT-135, HEIGHT-153, HEIGHT-153, HEIGHT-117, HEIGHT-117]).astype(jnp.int32),
lambda:jnp.array([HEIGHT-99,HEIGHT- 117,HEIGHT- 135,HEIGHT- 153,HEIGHT- 171,HEIGHT- 63,HEIGHT- 81, HEIGHT+20]).astype(jnp.int32),
lambda:jnp.array([HEIGHT-63, HEIGHT-81, HEIGHT-99, HEIGHT-117, HEIGHT-135, HEIGHT-153, HEIGHT-171, HEIGHT+20]).astype(jnp.int32),
lambda:jnp.array([HEIGHT-135, HEIGHT+20 ,  HEIGHT+20 ,  HEIGHT+20 ,  HEIGHT+20,  HEIGHT+20 , HEIGHT+20 , HEIGHT+20]).astype(jnp.int32),
]

MAX_PLAYER = 1
MAX_PLAYER_PROJECTILE = 1
MAX_PHOENIX = 8
MAX_BATS = 7
MAX_BOSS = 1
MAX_BOSS_BLOCK_GREEN = 1
MAX_BOSS_BLOCK_BLUE = 48
MAX_BOSS_BLOCK_RED = 104
SCORE_COLOR = (210, 210, 64)


# === GAME STATE ===
class PhoenixState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    step_counter: chex.Array
    enemies_x: chex.Array # Gegner X-Positionen
    enemies_y: chex.Array
    enemy_direction: chex.Array
    vertical_direction: chex.Array
    phoenix_cooldown: chex.Array
    projectile_x: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil
    projectile_y: chex.Array = jnp.array(-1)  # Standardwert: kein Projektil # Gegner Y-Positionen
    enemy_projectile_x: chex.Array = jnp.full((MAX_PHOENIX,), -1) # Enemy projectile X-Positionen
    enemy_projectile_y: chex.Array = jnp.full((MAX_PHOENIX,), -1) # Enemy projectile Y-Positionen

    score: chex.Array = jnp.array(0)  # Score
    lives: chex.Array = jnp.array(5) # Lives
    player_respawn_timer: chex.Array = 0 # Invincibility timer
    level: chex.Array = jnp.array(1)  # Level, starts at 1





class PhoenixOberservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_score: chex.Array
    lives: chex.Array

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
    player_sprites = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player.npy"))
    bg_sprites = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/pong/background.npy"))
    floor_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/floor.npy"))
    player_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/player_projectile.npy"))
    bat_high_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_high_wings.npy"))
    bat_low_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_low_wings.npy"))
    bat_2_high_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_high_wings.npy"))
    bat_2_low_wings_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/bats/bats_2_low_wings.npy"))
    enemy1_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix.npy"))
    enemy2_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_phoenix_2.npy"))
    boss_sprite = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/boss.npy"))
    enemy_projectile = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/enemy_projectile.npy"))


    SPRITE_PLAYER = jnp.expand_dims(player_sprites, axis=0)
    BG_SPRITE = jnp.expand_dims(np.zeros_like(bg_sprites), axis=0)
    SPRITE_FLOOR = jnp.expand_dims(floor_sprite, axis=0)
    SPRITE_PLAYER_PROJECTILE = jnp.expand_dims(player_projectile, axis=0)
    SPRITE_ENEMY1 = jnp.expand_dims(enemy1_sprite, axis=0)
    SPRITE_ENEMY2 = jnp.expand_dims(enemy2_sprite, axis=0)
    SPRITE_BAT_HIGH_WING = jnp.expand_dims(bat_high_wings_sprite, axis=0)
    SPRITE_BAT_LOW_WING = jnp.expand_dims(bat_low_wings_sprite, axis=0)
    SPRITE_BAT_2_HIGH_WING = jnp.expand_dims(bat_2_high_wings_sprite, axis=0)
    SPRITE_BAT_2_LOW_WING = jnp.expand_dims(bat_2_low_wings_sprite, axis=0)
    SPRITE_BOSS = jnp.expand_dims(boss_sprite, axis=0)
    SPRITE_ENEMY_PROJECTILE = jnp.expand_dims(enemy_projectile, axis=0)

    DIGITS = jr.load_and_pad_digits(os.path.join(MODULE_DIR, "./sprites/phoenix/digits/{}.npy"))
    LIFE_INDICATOR = jr.loadFrame(os.path.join(MODULE_DIR, "./sprites/phoenix/life_indicator.npy"))

    return (
        SPRITE_PLAYER,
        BG_SPRITE,
        SPRITE_PLAYER_PROJECTILE,
        SPRITE_FLOOR,
        SPRITE_ENEMY1,
        SPRITE_ENEMY2,
        SPRITE_BAT_HIGH_WING,
        SPRITE_BAT_LOW_WING,
        SPRITE_BAT_2_HIGH_WING,
        SPRITE_BAT_2_LOW_WING,
        SPRITE_BOSS,
        SPRITE_ENEMY_PROJECTILE,
        DIGITS,
        LIFE_INDICATOR,
    )
# load sprites on module layer
(SPRITE_PLAYER, SPRITE_BG, SPRITE_PLAYER_PROJECTILE, SPRITE_FLOOR, SPRITE_ENEMY1, SPRITE_ENEMY2, SPRITE_BAT_HIGH_WING, SPRITE_BAT_LOW_WING,SPRITE_BAT_2_HIGH_WING,SPRITE_BAT_2_LOW_WING,SPRITE_BOSS, SPRITE_ENEMY_PROJECTILE, DIGITS, LIFE_INDICATOR) = load_sprites()





@jax.jit
def player_step(
    state: PhoenixState, action: chex.Array) -> tuple[chex.Array]:

    step_size = 2 # Größerer Wert = schnellerer Schritt
    # left action
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
    # right action
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
    #movement right
    player_x = jnp.where(
        right, state.player_x + step_size, jnp.where(left, state.player_x - step_size, state.player_x)
    )
    # movement left
    player_x = jnp.where(
        player_x < PLAYER_BOUNDS[0], PLAYER_BOUNDS[0], jnp.where(player_x > PLAYER_BOUNDS[1], PLAYER_BOUNDS[1], player_x)
    )

    return player_x.astype(jnp.float32)


# Größe der Sprites
PROJECTILE_WIDTH = 2
PROJECTILE_HEIGHT = 4
ENEMY_WIDTH = 10
ENEMY_HEIGHT = 10


def phoenix_step(state):
    enemy_step_size = 0.4
    vertical_step_size = 0.3

    active_enemies = (state.enemies_x > -1) & (state.enemies_y < HEIGHT+10)

    # Prüfen, ob ein Gegner die linke oder rechte Grenze erreicht hat
    at_left_boundary = jnp.any(jnp.logical_and(state.enemies_x <= PLAYER_BOUNDS[0], active_enemies))
    at_right_boundary = jnp.any(jnp.logical_and(state.enemies_x >= PLAYER_BOUNDS[1] - ENEMY_WIDTH/2, active_enemies))
    # Richtung ändern, wenn eine Grenze erreicht wird
    new_direction = jax.lax.cond(
        at_left_boundary,
        lambda: jnp.full_like(state.enemy_direction, 1.0, dtype=jnp.float32),
        lambda: jax.lax.cond(
            at_right_boundary,
            lambda: jnp.full_like(state.enemy_direction, -1.0, dtype=jnp.float32),
            lambda: state.enemy_direction.astype(jnp.float32),
        ),
    )
    # Choose the two most bottom phoenix enemies to move
    jax.debug.print("Enemy Y positions: {}", state.enemies_y)
    jax.debug.print("ENEMY_Y: {}", ENEMY_POSITIONS_Y)
    enemy_indices = jnp.argsort(state.enemies_y, axis=0)[-2:]
    hit_bottom_mask = (state.enemies_y >= 159) & jnp.isin(jnp.arange(state.enemies_y.shape[0]), enemy_indices)
    new_vertical_direction = jnp.where(
        hit_bottom_mask,
        -state.vertical_direction,
        state.vertical_direction
    )
    new_enemies_y = jnp.where(
        jnp.isin(jnp.arange(state.enemies_y.shape[0]), enemy_indices),
        state.enemies_y + (new_vertical_direction * vertical_step_size),
        state.enemies_y
    )



    # Gegner basierend auf der Richtung bewegen, nur aktive Gegner
    new_enemies_x = jnp.where(active_enemies, state.enemies_x + (new_direction * enemy_step_size), state.enemies_x)

    # Begrenzung der Positionen innerhalb des Spielfelds
    new_enemies_x = jnp.clip(new_enemies_x, PLAYER_BOUNDS[0], PLAYER_BOUNDS[1])

    #jax.debug.print("dir :{}", new_vertical_direction)
    state = state._replace(
        enemies_x=new_enemies_x.astype(jnp.float32),
        enemy_direction=new_direction.astype(jnp.float32),
        enemies_y = new_enemies_y.astype(jnp.float32),
        vertical_direction = new_vertical_direction
    )

    # Aktualisierten Zustand zurückgeben
    return state

def bat_step(state):
    bat_step_size = 0.5
    active_bats = (state.enemies_x > -1) & (state.enemies_y < HEIGHT + 10)

    # Initialisiere neue Richtungen für jede Fledermaus
    new_directions = jnp.where(
        jnp.logical_and(state.enemies_x <= PLAYER_BOUNDS[0]+3, active_bats),
        jnp.ones(state.enemy_direction.shape, dtype=jnp.float32),  # Force array shape
        jnp.where(
            jnp.logical_and(state.enemies_x >= PLAYER_BOUNDS[1] - ENEMY_WIDTH / 2, active_bats),
            jnp.ones(state.enemy_direction.shape, dtype=jnp.float32) * -1,  # Force array shape
            state.enemy_direction.astype(jnp.float32)  # Ensure consistency
        )
    )

    # Bewege Fledermäuse basierend auf ihrer individuellen Richtung
    new_enemies_x = jnp.where(active_bats, state.enemies_x + (new_directions * bat_step_size), state.enemies_x)

    # Begrenze die Positionen innerhalb des Spielfelds
    new_enemies_x = jnp.clip(new_enemies_x, PLAYER_BOUNDS[0], PLAYER_BOUNDS[1])
    state = state._replace(
        enemies_x=new_enemies_x.astype(jnp.float32),
        enemies_y=state.enemies_y.astype(jnp.float32),
        enemy_direction=new_directions.astype(jnp.float32)
    )

    return state



class JaxPhoenix(JaxEnvironment[PhoenixState, PhoenixOberservation, PhoenixInfo, None]):
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: PhoenixState) -> PhoenixOberservation:
        player = EntityPosition(x=state.player_x, y=state.player_y)
        return PhoenixOberservation(
            player_x = player[0],
            player_y= player[1],
            player_score = state.score,
            lives= state.lives
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: PhoenixState, all_rewards: jnp.ndarray) -> PhoenixInfo:
        return PhoenixInfo(
            step_counter=0,
        )

    def action_space(self) -> Space:
        return self._get_observation(PhoenixState)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: PhoenixState) -> Tuple[bool, PhoenixState]:
        return jnp.less_equal(state.lives,0)
    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))
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
        ]# Add step counter tracking
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(42)) -> Tuple[PhoenixOberservation, PhoenixState]:

        # Initialisierung der Gegnerpositionen
        enemy_spawn_x = ENEMY_POSITIONS_X
        enemy_spawn_y = ENEMY_POSITIONS_Y

        return_state = PhoenixState(
            player_x=jnp.array(PLAYER_POSITION[0]),
            player_y=jnp.array(PLAYER_POSITION[1]),
            step_counter=jnp.array(0),
            enemies_x = enemy_spawn_x.astype(jnp.float32),
            enemies_y = enemy_spawn_y.astype(jnp.float32),
            enemy_direction =  jnp.full((8,), -1.0),
            enemy_projectile_x=jnp.full((MAX_PHOENIX,), -1),
            enemy_projectile_y=jnp.full((MAX_PHOENIX,), -1),
            projectile_x=jnp.array(-1),  # Standardwert: kein Projektil
            score = jnp.array(0), # Standardwert: Score=0
            lives=jnp.array(5), # Standardwert: 5 Leben
            player_respawn_timer=jnp.array(5),
            level=jnp.array(1),
            phoenix_cooldown=jnp.array(30),
            vertical_direction=jnp.full((8,),1.0)
        )

        initial_obs = self._get_observation(return_state)
        return initial_obs, return_state


    def step(self,state, action: Action) -> Tuple[PhoenixOberservation, PhoenixState, float, bool, PhoenixInfo]:
        player_x = player_step(state, action)

        can_fire = state.projectile_y < 0
        projectile_x = jnp.where((action == Action.FIRE) & can_fire, state.player_x + 2, state.projectile_x)
        projectile_y = jnp.where((action == Action.FIRE) & can_fire, state.player_y - 1, state.projectile_y - 5) # durch das -3 wird das Projektil schneller (Projektil geschwindigkeit)

        # Projektil entfernen, wenn es obere Grenze erreicht:
        projectile_y = jnp.where(projectile_y < 0, -6, projectile_y)




        # Move enemies
        #enemies_x, enemies_y,enemy_direction = jax.lax.cond(
        #    jnp.logical_or((state.level%5) == 1, (state.level%5) == 2),
        #    lambda: phoenix_step(state),
        #    lambda: jax.lax.cond(
        #        jnp.logical_or((state.level%5) == 3, (state.level%5) == 4),
        #        lambda: bat_step(state),
        #        lambda: (state.enemies_x.astype(jnp.float32), state.enemies_y.astype(jnp.float32),jnp.broadcast_to(state.enemy_direction.astype(jnp.float32),(8,)))  # No movement for level 5
        #    )
        #)

        state = jax.lax.cond(
            jnp.logical_or((state.level % 5) == 1, (state.level % 5) == 2),
            lambda: phoenix_step(state),
            lambda: jax.lax.cond(
                jnp.logical_or((state.level % 5) == 3, (state.level % 5) == 4),
                lambda: bat_step(state),
                lambda: state
            )
        )
        ###Enemy shooting
        # use step_counter for randomness
        def generate_fire_key_and_chance(step_counter: int, fire_chance: float) -> Tuple[jax.random.PRNGKey, float]:
            key = jax.random.PRNGKey(step_counter)
            return key, fire_chance

        key, fire_chance = generate_fire_key_and_chance(state.step_counter, 0.0075)  # 2% chance per enemy per frame

        # Random decision: should each enemy fire?
        enemy_should_fire = jax.random.uniform(key, (MAX_PHOENIX,)) < fire_chance

        # Fire only from active enemies
        can_fire = (state.enemy_projectile_y < 0) & (state.enemies_x > -1)
        enemy_fire_mask = enemy_should_fire & can_fire

        # Fire from current enemy positions
        enemy_projectile_x = jnp.where(enemy_fire_mask, state.enemies_x + ENEMY_WIDTH // 2,
                                           state.enemy_projectile_x)
        enemy_projectile_y = jnp.where(enemy_fire_mask, state.enemies_y + ENEMY_HEIGHT, state.enemy_projectile_y)

        # Move enemy projectiles downwards
        enemy_projectile_y = jnp.where(state.enemy_projectile_y >= 0, state.enemy_projectile_y + 4, # +4 regelt enemy projectile speed
                                           enemy_projectile_y)

        # Remove enemy projectile if off-screen
        enemy_projectile_y = jnp.where(enemy_projectile_y > 185 - PROJECTILE_HEIGHT, -1, enemy_projectile_y)


        projectile_pos = jnp.array([projectile_x, projectile_y])
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)

        def check_collision(entity_pos, projectile_pos):
            enemy_x, enemy_y = entity_pos
            projectile_x, projectile_y = projectile_pos

            collision_x = (projectile_x + PROJECTILE_WIDTH > enemy_x) & (projectile_x < enemy_x + ENEMY_WIDTH)
            collision_y = (projectile_y + PROJECTILE_HEIGHT > enemy_y) & (projectile_y < enemy_y + ENEMY_HEIGHT)
            return collision_x & collision_y


        # Kollisionsprüfung Gegner
        enemy_collisions = jax.vmap(lambda enemy_pos: check_collision(enemy_pos, projectile_pos))(enemy_positions)
        enemy_hit_detected = jnp.any(enemy_collisions)


        # Gegner und Projektil entfernen wenn eine Kollision erkannt wurde
        enemies_x = jnp.where(enemy_collisions, -1, state.enemies_x).astype(jnp.float32)
        enemies_y = jnp.where(enemy_collisions, HEIGHT+20, state.enemies_y).astype(jnp.float32)
        projectile_x = jnp.where(enemy_hit_detected, -1, projectile_x)
        projectile_y = jnp.where(enemy_hit_detected, -1, projectile_y)
        score = jnp.where(enemy_hit_detected, state.score + 20, state.score)

        # Checken ob alle Gegner getroffen wurden
        all_enemies_hit = jnp.all(enemies_y >= HEIGHT + 10)
        #jax.debug.print("All enemies hit: {}", all_enemies_hit)
        #jax.debug.print("Enemies X: {}", enemies_x)
        #jax.debug.print("Enemies Y: {}", enemies_y)
        new_level = jnp.where(all_enemies_hit, (state.level % 5) + 1, state.level)
        new_enemies_x = jax.lax.cond(
            all_enemies_hit,
            lambda: jax.lax.switch((new_level -1 )% 5, ENEMY_POSITIONS_X_LIST).astype(jnp.float32),
            lambda: state.enemies_x.astype(jnp.float32)
        )
        new_enemies_y = jax.lax.cond(
            all_enemies_hit,
            lambda: jax.lax.switch((new_level -1 )% 5, ENEMY_POSITIONS_Y_LIST).astype(jnp.float32),
            lambda: enemies_y.astype(jnp.float32)
        )
        enemies_x = new_enemies_x
        enemies_y = new_enemies_y
        level = new_level

        #jax.debug.print("Level: {}",level )
        def check_player_hit(projectile_xs, projectile_ys, player_x, player_y):
            def is_hit(px, py):
                hit_x = (px + PROJECTILE_WIDTH > player_x) & (px < player_x + 5)
                hit_y = (py + PROJECTILE_HEIGHT > player_y) & (py < player_y + PROJECTILE_HEIGHT)
                return hit_x & hit_y

            hits = jax.vmap(is_hit)(projectile_xs, projectile_ys)
            return jnp.any(hits)



        # Kollisionsüberprüfung Spieler
        # Remaining lives updaten und Spieler neu Spawnen
        is_vulnerable = state.player_respawn_timer <= 0
        player_hit_detected = jnp.where(is_vulnerable, check_player_hit(state.enemy_projectile_x, enemy_projectile_y, player_x, state.player_y), False)
        lives = jnp.where(player_hit_detected, state.lives - 1, state.lives)
        player_x = jnp.where(player_hit_detected, PLAYER_POSITION[0], player_step(state, action))
        player_respawn_timer = jnp.where(
            player_hit_detected,
            5,
            jnp.maximum(state.player_respawn_timer - 1, 0)
        )
        # Respawn remaining enemies
        enemy_respawn_x = jax.lax.switch((level -1) % 5, ENEMY_POSITIONS_X_LIST).astype(jnp.float32)
        enemy_respawn_y = jax.lax.switch((level - 1) % 5, ENEMY_POSITIONS_Y_LIST).astype(jnp.int32)

        enemy_respawn_mask = jnp.logical_and(player_hit_detected, (enemies_x > 0) & (enemies_y < HEIGHT + 10))
        enemies_x = jnp.where(enemy_respawn_mask, enemy_respawn_x, enemies_x)
        enemies_y = jnp.where(enemy_respawn_mask, enemy_respawn_y, enemies_y)



        # Enemy Projectile entfernen wenn eine Kollision mit dem Spieler erkannt wurde
        enemy_projectile_x = jnp.where(player_hit_detected, -1, enemy_projectile_x)
        enemy_projectile_y = jnp.where(player_hit_detected, -1, enemy_projectile_y)

        return_state = PhoenixState(
            player_x = player_x,
            player_y = state.player_y,
            step_counter = state.step_counter + 1,
            projectile_x = projectile_x,
            projectile_y = projectile_y,
            enemies_x = enemies_x,
            enemies_y = enemies_y,
            enemy_direction = state.enemy_direction,
            score= score,
            enemy_projectile_x=enemy_projectile_x,
            enemy_projectile_y=enemy_projectile_y,
            lives=lives,
            player_respawn_timer = player_respawn_timer,
            level = level,
            phoenix_cooldown = state.phoenix_cooldown,
            vertical_direction=state.vertical_direction
        )
        observation = self._get_observation(return_state)
        env_reward = jnp.where(enemy_hit_detected, 1.0, 0.0)
        done = self._get_done(return_state)
        info = self._get_info(return_state, env_reward)
        return observation, return_state, env_reward, done, info

from jaxatari.renderers import JAXGameRenderer

class PhoenixRenderer(JAXGameRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jr.create_initial_frame(width=WIDTH, height=HEIGHT)

        # Render background
        frame_bg = jr.get_sprite_frame(SPRITE_BG, 0)
        raster = jr.render_at(raster, 0, 0, frame_bg)
        # Render floor
        frame_floor = jr.get_sprite_frame(SPRITE_FLOOR, 0)
        raster = jr.render_at(raster, 0, 185, frame_floor)
        # Render player
        frame_player = jr.get_sprite_frame(SPRITE_PLAYER, 0)
        raster = jr.render_at(raster, state.player_x, state.player_y, frame_player)
        # Render projectile
        frame_projectile = jr.get_sprite_frame(SPRITE_PLAYER_PROJECTILE, 0)
        # Render enemies
        frame_enemy_1 = jr.get_sprite_frame(SPRITE_ENEMY1, 0)
        frame_enemy_2 = jr.get_sprite_frame(SPRITE_ENEMY2, 0)
        frame_bat_high_wings = jr.get_sprite_frame(SPRITE_BAT_HIGH_WING, 0)
        frame_bat_low_wings = jr.get_sprite_frame(SPRITE_BAT_LOW_WING, 0)
        frame_bat_2_high_wings = jr.get_sprite_frame(SPRITE_BAT_2_HIGH_WING, 0)
        frame_bat_2_low_wings = jr.get_sprite_frame(SPRITE_BAT_2_LOW_WING, 0)
        frame_enemy_projectile = jr.get_sprite_frame(SPRITE_ENEMY_PROJECTILE, 0)
        frame_boss = jr.get_sprite_frame(SPRITE_BOSS, 0)


        def render_enemy(raster, enemy_pos):
            x, y = enemy_pos

            def render_level1(r):
                return jr.render_at(r, x, y, frame_enemy_1)

            def render_level2(r):
                return jr.render_at(r, x, y, frame_enemy_2)
            def render_level3(r):
                return jr.render_at(r, x, y, frame_bat_high_wings)
            def render_level4(r):
                return jr.render_at(r, x, y, frame_bat_2_high_wings)
            def render_level5(r):
                return jr.render_at(r, x, y, frame_boss)

            def render_if_active(r):
                return jax.lax.switch(
                    state.level - 1,
                    [
                        render_level1,
                        render_level2,
                        render_level3,
                        render_level4,
                        render_level5,
                    ],
                    r
                )

            raster = jax.lax.cond(x > -1, render_if_active, lambda r: r, raster)

            return raster, None
        enemy_positions = jnp.stack((state.enemies_x, state.enemies_y), axis=1)
        raster, _ = jax.lax.scan(render_enemy, raster, enemy_positions)

        # Render player projectiles
        def render_player_projectile(r):
            return jr.render_at(r, state.projectile_x, state.projectile_y, frame_projectile)

        raster = jax.lax.cond(
            state.projectile_x > -1,
            render_player_projectile,
            lambda r: r,
            raster
        )

        def render_enemy_projectile(raster, projectile_pos):
            x, y = projectile_pos
            return jax.lax.cond(
                y > -1,
                lambda r: jr.render_at(r, x, y, frame_enemy_projectile),
                lambda r: r,
                raster
            ), None

        # render enemy projectiles
        enemy_proj_positions = jnp.stack((state.enemy_projectile_x, state.enemy_projectile_y), axis=1)
        raster, _ = jax.lax.scan(render_enemy_projectile, raster, enemy_proj_positions)
        # render score
        score_array = jr.int_to_digits(state.score, max_digits=5)  # 5 for now
        raster = jr.render_label(raster, 60, 10, score_array, DIGITS, spacing=8)
        # render lives
        lives_value = jnp.sum(jr.int_to_digits(state.lives, max_digits=2))
        raster = jr.render_indicator(raster, 70, 20, lives_value, LIFE_INDICATOR, spacing=4)

        return raster




