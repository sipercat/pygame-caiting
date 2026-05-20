import pygame
import random
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import os
from abc import ABC, abstractmethod

pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Enhanced Shooter - High Intensity Learning")
clock = pygame.time.Clock()

font = pygame.font.SysFont("malgungothic", 20)
small_font = pygame.font.SysFont("malgungothic", 15)
tiny_font = pygame.font.SysFont("malgungothic", 10)
big_font = pygame.font.SysFont("malgungothic", 40)

WHITE, BLACK, BLUE, RED, GOLD, GREEN, GRAY, PURPLE = (245, 245, 245), (30, 30, 30), (70, 130, 255), (220, 80, 80), (255, 210, 0), (80, 200, 120), (180, 180, 180), (148, 0, 211)

SIM_TIME = 0.0

def get_time(): return SIM_TIME

def format_time(ms_time):
    seconds = (int(ms_time) // 1000) % 60
    minutes = int(ms_time) // 60000
    millis = (int(ms_time) % 1000) // 10
    return f"{minutes:02d}:{seconds:02d}.{millis:02d}"

# --- [RL 모델 및 에이전트 (학습 강도 대폭 업그레이드)] ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [NEW] 로컬 모델과 타겟 모델 분리 (학습 안정성 및 속도 극대화)
        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # 타겟 모델은 평가용으로만 사용
        
        # [MODIFIED] 학습률(LR) 2배 증가로 더 강하게 학습
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = collections.deque(maxlen=50000)
        
        # [MODIFIED] 배치 사이즈 상향 (64 -> 128)
        self.batch_size = 128 
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        # [MODIFIED] 엡실론 감쇠 가속화 (무작위 탐험을 더 빨리 끝냄)
        self.epsilon_decay = 0.995 
        
        # [NEW] 타겟 네트워크 소프트 업데이트 비율
        self.tau = 0.005 
        self.loss = 0

    def save(self):
        torch.save(self.model.state_dict(), "../shooter_model_full.pth")
        
    def load(self):
        if os.path.exists("../shooter_model_full.pth"):
            self.model.load_state_dict(torch.load("../shooter_model_full.pth", map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict()) # 로드 시 동기화
            return True
        return False

    def get_heuristic_action(self, env, last_action):
        player = env.player
        bosses = [e for e in env.objects if isinstance(e, Boss) and e.is_alive]
        boss = min(bosses, key=lambda b: math.hypot(b.x - player.x, b.y - player.y)) if bosses else None
        boss_dist = math.hypot(boss.x - player.x, boss.y - player.y) if boss else 9999

        if boss_dist < 320:
            best_act = 0
            max_score = -float('inf')
            moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            
            for act, (dx, dy) in moves.items():
                nx = player.x + dx * player.speed * 15
                ny = player.y + dy * player.speed * 15
                b_dist = math.hypot(boss.x - nx, boss.y - ny)
                
                wall_penalty = 0
                margin = 120
                if nx < margin: wall_penalty += (margin - nx) * 20
                if nx > WIDTH - margin: wall_penalty += (nx - (WIDTH - margin)) * 20
                if ny < margin: wall_penalty += (margin - ny) * 20
                if ny > HEIGHT - margin: wall_penalty += (ny - (HEIGHT - margin)) * 20
                
                score = b_dist - wall_penalty
                if score > max_score:
                    max_score = score; best_act = act
            return best_act

        if player.hp / player.max_hp <= 0.4:
            heals = [h for h in env.objects if isinstance(h, HealPack)]
            if heals:
                nearest_heal = min(heals, key=lambda h: math.hypot(h.x - player.x, h.y - player.y))
                dx = nearest_heal.x - player.x
                dy = nearest_heal.y - player.y
                if abs(dx) > abs(dy): return 2 if dx > 0 else 1
                else: return 4 if dy > 0 else 3

        if get_time() >= player.next_shoot_time and boss_dist >= 350:
            return 5
            
        enemies = [e for e in env.objects if isinstance(e, Enemy) and e.is_alive]
        if not enemies: return random.choice([1, 2, 3, 4])
        
        best_action = 0
        max_safety = -float('inf')
        moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
        for act, (dx, dy) in moves.items():
            nx = player.x + dx * player.speed * 15
            ny = player.y + dy * player.speed * 15
            wall_penalty = 0
            if nx < 50 or nx > WIDTH - 50: wall_penalty += 800
            if ny < 50 or ny > HEIGHT - 50: wall_penalty += 800
            
            future_dists = [math.hypot((e.x + e.vx * 15) - nx, (e.y + e.vy * 15) - ny) for e in enemies]
            min_enemy_dist = min(future_dists) if future_dists else 999
            safety_score = min_enemy_dist - wall_penalty
            if safety_score > max_safety:
                max_safety = safety_score; best_action = act
                
        return best_action

    def act(self, state, env, last_action=0, train=True, pending_level_up=False):
        valid_actions = [6, 7, 8] if pending_level_up else [0, 1, 2, 3, 4, 5]
        
        if train and random.random() < self.epsilon:
            if not pending_level_up:
                if random.random() < 0.85: 
                    return self.get_heuristic_action(env, last_action)
                return random.choice(valid_actions)
            return random.choice(valid_actions)
            
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).squeeze()
        
        if not train and not pending_level_up:
            if q_values.max().item() < 0.5:
                return self.get_heuristic_action(env, last_action)

        valid_q_values = {a: q_values[a].item() for a in valid_actions}
        if not valid_q_values: return 0
        return max(valid_q_values, key=valid_q_values.get)

    def train_step(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(np.array(a)).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(np.array(r)).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(np.array(d)).unsqueeze(1).to(self.device)
        
        # 현재 Q-value 계산
        curr_q = self.model(s).gather(1, a)
        
        # [NEW] 타겟 네트워크를 사용해 다음 상태의 최대 Q-value 예측
        with torch.no_grad():
            next_q = self.target_model(ns).max(1)[0].unsqueeze(1)
        target_q = r + (self.gamma * next_q * (1 - d))
        
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # [NEW] 그래디언트 클리핑: 학습률이 높아졌을 때 모델이 터지는 것을 방지
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.loss = loss.item()
        
        # [NEW] 타겟 네트워크 소프트 업데이트 (서서히 로컬 모델을 따라가게 함)
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# --- [이하 게임 객체 및 환경 코드는 동일] ---
class GameObject(ABC):
    def __init__(self): self._alive = True
    @property
    def is_alive(self): return self._alive
    @property
    def is_dead(self): return not self._alive
    def destroy(self): self._alive = False
    @property
    def can_collide_with_player(self): return False
    @property
    def can_collide_with_bullet(self): return False
    @property
    def is_bullet(self): return False
    @property
    @abstractmethod
    def rect(self): pass
    @abstractmethod
    def update(self): pass
    @abstractmethod
    def draw(self, surface): pass
    def on_player_collision(self, player): pass
    def on_bullet_collision(self, bullet, player): pass

class LivingEntity(GameObject):
    def __init__(self, x, y, radius, max_hp):
        super().__init__()
        self._x, self._y, self._radius, self._max_hp = x, y, radius, float(max_hp)
        self._hp = self._max_hp
        self._debuffs = []
        self.vx = 0.0 
        self.vy = 0.0
    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def radius(self): return self._radius
    @property
    def max_hp(self): return self._max_hp
    @property
    def hp(self): return self._hp
    @hp.setter
    def hp(self, value): self._hp = max(0.0, min(self._max_hp, float(value)))
    @property
    def rect(self): return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)
    def take_damage(self, amount, killer=None):
        if not self.is_alive: return
        self.hp -= amount
        if self.hp <= 0: self.die(killer)
    def die(self, killer=None): self.destroy()
    def apply_debuff(self, dps, duration, killer):
        self._debuffs.append({"dps": dps, "time": duration, "killer": killer})
    def update_debuffs(self, dt):
        alive_debuffs = []
        for db in self._debuffs:
            self.take_damage(db["dps"] * dt, db["killer"])
            db["time"] -= dt
            if db["time"] > 0 and self.is_alive: alive_debuffs.append(db)
        self._debuffs = alive_debuffs
    def draw_hp_bar(self, surface, y_offset=-8):
        if self._max_hp > 1 or isinstance(self, Enemy):
            bar_w, bar_h = self._radius * 2, 5
            bar_x, bar_y = int(self._x - self._radius), int(self._y - self._radius + y_offset)
            pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_w, bar_h))
            current_w = bar_w * (self._hp / self._max_hp)
            if current_w > 0:
                bar_color = GREEN if isinstance(self, Enemy) else RED
                pygame.draw.rect(surface, bar_color, (bar_x, bar_y, current_w, bar_h))
            pygame.draw.rect(surface, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)

class Player(LivingEntity):
    def __init__(self):
        super().__init__(x=100, y=300, radius=20, max_hp=100)
        self._speed = 2.5
        self._attack_power = 5.0
        self._score = 0
        self.__freeze_end_time = 0
        self.__attack_cooldown = 400.0
        self.__next_shoot_time = 0
        self.pending_level_ups = 0
        self.atk_lvl, self.spd_lvl, self.mov_lvl = 0, 0, 0
        self.dash_ready = False
        self.bonus_bullet_pct = 0.0

    @property
    def score(self): return self._score
    @property
    def attack_power(self): return self._attack_power
    @property
    def attack_cooldown(self): return self.__attack_cooldown
    @property
    def speed(self): return self._speed
    @property
    def next_shoot_time(self): return self.__next_shoot_time
    @property
    def freeze_end_time(self): return self.__freeze_end_time

    def add_score(self, amount):
        if amount >= 0: self._score += amount
    def refill_dash(self):
        if self.mov_lvl >= 10: self.dash_ready = True
    def dash(self, mouse_pos):
        if self.mov_lvl >= 10 and self.dash_ready:
            mx, my = mouse_pos
            dx, dy = mx - self._x, my - self._y
            dist = math.hypot(dx, dy)
            if dist > 0:
                dash_dist = 150
                self._x = max(self._radius, min(WIDTH - self._radius, self._x + (dx / dist) * dash_dist))
                self._y = max(self._radius, min(HEIGHT - self._radius, self._y + (dy / dist) * dash_dist))
            self.dash_ready = False
    def upgrade_attack_power(self):
        self.atk_lvl += 1
        bonus_pct = 50.0 / math.sqrt(self.atk_lvl)
        self._attack_power *= (1.0 + (bonus_pct / 100.0))
    def upgrade_attack_speed(self):
        self.spd_lvl += 1
        target_cooldown = self.__attack_cooldown * 0.75
        if target_cooldown < 100.0:
            if self.__attack_cooldown > 100.0:
                applied_pct = ((self.__attack_cooldown - 100.0) / self.__attack_cooldown) * 100.0
                excess_pct = 25.0 - applied_pct
                self.__attack_cooldown = 100.0
            else: excess_pct = 25.0
            if excess_pct > 0: self.bonus_bullet_pct += excess_pct * 2.0 / math.sqrt(self.spd_lvl)
        else: self.__attack_cooldown = target_cooldown
    def upgrade_move_speed(self):
        self.mov_lvl += 1
        self._speed += 0.5
        if self.mov_lvl >= 10: self.dash_ready = True
    def shoot(self, objects_ref, is_auto=False):
        current_time = get_time()
        if current_time < self.__next_shoot_time: return []
        effective_cooldown = 200.0 if is_auto else self.__attack_cooldown
        self.__next_shoot_time = current_time + effective_cooldown
        self.__freeze_end_time = current_time + effective_cooldown * 0.375
        bullets = []
        total_bullets = 1 + int(self.bonus_bullet_pct // 100)
        gap = 12
        start_y = self._y - (total_bullets - 1) * gap / 2.0
        for i in range(total_bullets):
            bullets.append(Bullet(self._x + self._radius, start_y + i * gap, objects_ref, self))
        return bullets

    def update(self, ai_action=None):
        if not self.is_alive: return
        accel = self._speed * 0.6  
        friction = 0.8 
        if get_time() >= self.__freeze_end_time:
            if ai_action is not None:
                if ai_action == 1: self.vx -= accel
                elif ai_action == 2: self.vx += accel
                elif ai_action == 3: self.vy -= accel
                elif ai_action == 4: self.vy += accel
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:  self.vx -= accel
                if keys[pygame.K_RIGHT]: self.vx += accel
                if keys[pygame.K_UP]:    self.vy -= accel
                if keys[pygame.K_DOWN]:  self.vy += accel

        self.vx *= friction; self.vy *= friction
        max_speed = self._speed * 1.5
        current_speed = math.hypot(self.vx, self.vy)
        if current_speed > max_speed:
            self.vx = (self.vx / current_speed) * max_speed
            self.vy = (self.vy / current_speed) * max_speed

        self._x = max(self._radius, min(WIDTH - self._radius, self._x + self.vx))
        self._y = max(self._radius, min(HEIGHT - self._radius, self._y + self.vy))

    def draw(self, surface):
        color = GRAY if not self.is_alive else BLUE
        pygame.draw.circle(surface, color, (int(self._x), int(self._y)), self._radius)

class Enemy(LivingEntity):
    def __init__(self, player):
        super().__init__(x=0, y=0, radius=20, max_hp=2)
        self.player = player
        self._speed = random.choice([3, 4, 5, 6])
        self.reset()
    @property
    def can_collide_with_player(self): return True
    @property
    def can_collide_with_bullet(self): return True
    def reset(self):
        self._alive = True
        self._x = WIDTH + self._radius
        self._y = random.randint(50, HEIGHT - 50)
        self._max_hp = float(2 + (self.player.score // 200))
        self.hp = self._max_hp
        self._debuffs.clear()
        self.vx = -self._speed
        self.vy = 0.0
    def die(self, killer=None):
        if killer: killer.add_score(200)
        self.reset()
    def update(self):
        self.update_debuffs(1 / 60.0)
        self.vx = -self._speed
        self._x += self.vx
        if self._x < -self._radius: self.reset()
    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self._x), int(self._y)), self._radius)
        self.draw_hp_bar(surface)
    def on_player_collision(self, player): player.take_damage(1)
    def on_bullet_collision(self, bullet, player):
        if self.is_alive and bullet.is_alive:
            self.take_damage(player.attack_power, player)
            bullet.destroy()

class Boss(LivingEntity):
    def __init__(self, spawn_count, player):
        super().__init__(x=0, y=0, radius=40, max_hp=int(75 * math.sqrt(spawn_count)))
        self.player = player
        self._speed = 1.1 * math.sqrt(spawn_count)
        while True:
            self._x, self._y = random.randint(self._radius, WIDTH - self._radius), random.randint(self._radius, HEIGHT - self._radius)
            if math.hypot(self._x - self.player.x, self._y - self.player.y) >= 400: break
    @property
    def can_collide_with_player(self): return True
    @property
    def can_collide_with_bullet(self): return True
    def die(self, killer=None):
        if killer:
            killer.add_score(200)
            killer.pending_level_ups += 1
        super().die(killer)
    def update(self):
        if not self.is_alive: return
        self.update_debuffs(1 / 60.0)
        dx, dy = self.player.x - self._x, self.player.y - self._y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.vx = (dx / dist) * self._speed
            self.vy = (dy / dist) * self._speed
            self._x += self.vx
            self._y += self.vy
    def draw(self, surface):
        pygame.draw.circle(surface, PURPLE, (int(self._x), int(self._y)), self._radius)
        self.draw_hp_bar(surface, y_offset=5)
    def on_player_collision(self, player): player.take_damage(2)
    def on_bullet_collision(self, bullet, player):
        if self.is_alive and bullet.is_alive:
            self.take_damage(player.attack_power, player)
            bullet.destroy()

class Coin(GameObject):
    def __init__(self):
        super().__init__()
        self._radius, self._dy, self._direction = 12, 1, 1
        self.reset()
    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def can_collide_with_player(self): return True
    @property
    def rect(self): return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)
    def reset(self):
        self._x, self._y = random.randint(100, WIDTH - 50), random.randint(50, HEIGHT - 50)
    def update(self):
        self._y += self._dy * self._direction
        if self._y > HEIGHT - 30 or self._y < 30: self._direction *= -1
    def draw(self, surface):
        pygame.draw.circle(surface, GOLD, (int(self._x), int(self._y)), self._radius)
        pygame.draw.circle(surface, BLACK, (int(self._x), int(self._y)), self._radius, 2)
    def on_player_collision(self, player):
        player.add_score(10)
        self.reset()

class HealPack(GameObject):
    def __init__(self):
        super().__init__()
        self._radius, self._heal_amount = 20, 20
        self.reset()
    @property
    def x(self): return self._x
    @property
    def y(self): return self._y
    @property
    def can_collide_with_player(self): return True
    @property
    def rect(self): return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)
    def reset(self):
        self._x, self._y = random.randint(100, WIDTH - 50), random.randint(50, HEIGHT - 50)
    def update(self): pass
    def draw(self, surface):
        pygame.draw.circle(surface, GREEN, (int(self._x), int(self._y)), self._radius)
    def on_player_collision(self, player):
        player.hp += self._heal_amount
        self.reset()

class Bullet(GameObject):
    def __init__(self, x, y, objects_ref, player_ref):
        super().__init__()
        self._x, self._y, self._speed, self._radius = x, y, 10, 5
        self._objects, self._player = objects_ref, player_ref
        self._dx = self._speed
        self._dy = 0
    @property
    def is_bullet(self): return True
    @property
    def rect(self): return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)
    def update(self):
        target, min_dist = None, float('inf')
        for obj in self._objects:
            if obj.is_alive and (isinstance(obj, Enemy) or isinstance(obj, Boss)):
                dist = math.hypot(obj.rect.centerx - self._x, obj.rect.centery - self._y)
                if dist < min_dist: min_dist, target = dist, obj
        if target and min_dist > 0:
            self._dx = ((target.rect.centerx - self._x) / min_dist) * self._speed
            self._dy = ((target.rect.centery - self._y) / min_dist) * self._speed
        self._x += self._dx
        self._y += self._dy
        if self._x < -10 or self._x > WIDTH + 10 or self._y < -10 or self._y > HEIGHT + 10: self.destroy()
    def draw(self, surface):
        pygame.draw.circle(surface, BLACK, (int(self._x), int(self._y)), self._radius)

class GameEnv:
    def __init__(self):
        self.record_100k_time = None
        self.play_time_ms = 0
        self.prev_boss_exists = False
        self.prev_boss_dist = None
        self.last_action = 0 
        self.reset()
        
    def reset(self):
        global SIM_TIME
        SIM_TIME = 0.0
        self.player = Player()
        self.objects = [self.player]
        for _ in range(6): self.objects.append(Enemy(self.player))
        for _ in range(10): self.objects.append(Coin())
        for _ in range(2): self.objects.append(HealPack())
        self.boss_spawn_count = 0
        self.prev_hp = self.player.hp
        self.prev_score = self.player.score
        self.record_100k_time = None
        self.play_time_ms = 0
        self.prev_boss_exists = False
        self.prev_boss_dist = None
        self.last_action = 0 
        return self.get_state()
    
    def get_state(self):
        current_tick = get_time()
        state = [
            self.player.x / WIDTH, (WIDTH - self.player.x) / WIDTH,
            self.player.y / HEIGHT, (HEIGHT - self.player.y) / HEIGHT,
            max(0.0, (self.player.next_shoot_time - current_tick) / self.player.attack_cooldown),
            self.player.attack_cooldown / 400.0,
            self.player.hp / self.player.max_hp,
            min(self.player.atk_lvl / 10.0, 1.0),
            min(self.player.spd_lvl / 10.0, 1.0),
            min(self.player.mov_lvl / 10.0, 1.0),
            1.0 if current_tick < self.player.freeze_end_time else 0.0,
            1.0 if self.player.pending_level_ups > 0 else 0.0,
            self.player.vx / 10.0, self.player.vy / 10.0
        ]
        
        bosses = [o for o in self.objects if isinstance(o, Boss) and o.is_alive]
        if bosses:
            b = bosses[0]
            state.extend([(b.x - self.player.x) / WIDTH, (b.y - self.player.y) / HEIGHT, b.hp / b.max_hp, b.vx / 10.0, b.vy / 10.0])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
        enemies = [o for o in self.objects if isinstance(o, Enemy) and o.is_alive]
        enemies.sort(key=lambda e: math.hypot(e.x - self.player.x, e.y - self.player.y))
        for i in range(3):
            if i < len(enemies):
                e = enemies[i]
                state.extend([(e.x - self.player.x) / WIDTH, (e.y - self.player.y) / HEIGHT, e.vx / 10.0, e.vy / 10.0])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
                
        coins = [o for o in self.objects if isinstance(o, Coin)]
        heals = [o for o in self.objects if isinstance(o, HealPack)]
        coins.sort(key=lambda c: math.hypot(c.x - self.player.x, c.y - self.player.y))
        heals.sort(key=lambda h: math.hypot(h.x - self.player.x, h.y - self.player.y))
        state.extend([(coins[0].x - self.player.x) / WIDTH, (coins[0].y - self.player.y) / HEIGHT] if coins else [0.0, 0.0])
        state.extend([(heals[0].x - self.player.x) / WIDTH, (heals[0].y - self.player.y) / HEIGHT] if heals else [0.0, 0.0])
        return np.array(state, dtype=np.float32)

    def step(self, action):
        reward = 0.0
        done = False
        
        if self.player.pending_level_ups > 0:
            if action == 6: self.player.upgrade_attack_power(); self.player.pending_level_ups -= 1
            elif action == 7: self.player.upgrade_attack_speed(); self.player.pending_level_ups -= 1
            elif action == 8: self.player.upgrade_move_speed(); self.player.pending_level_ups -= 1
            else: return self.get_state(), 0, False
            return self.get_state(), 10.0, done

        dt_ms = 1000.0 / 60.0
        global SIM_TIME
        SIM_TIME += dt_ms
        self.play_time_ms += dt_ms
        
        if self.player.score >= 100000 and self.record_100k_time is None:
            self.record_100k_time = format_time(self.play_time_ms)

        if action in [1, 2, 3, 4] and self.last_action in [1, 2, 3, 4]:
            if (action == 1 and self.last_action == 2) or (action == 2 and self.last_action == 1) or \
               (action == 3 and self.last_action == 4) or (action == 4 and self.last_action == 3):
                reward -= 10.0 
                
        if action in [1, 2, 3, 4]:
            self.last_action = action

        shot_fired = False
        if action == 5:
            if get_time() >= self.player.next_shoot_time: shot_fired = True
                
        self.player.update(ai_action=action)
        
        bosses_alive = [o for o in self.objects if isinstance(o, Boss)]
        
        if action == 5:
            bullets = self.player.shoot(self.objects, is_auto=(self.player.spd_lvl >= 10))
            if bullets: self.objects.extend(bullets)
            if shot_fired and reward > -10.0: reward += 10.0 
            elif not shot_fired: reward -= 1.0  

        if self.player.score >= (self.boss_spawn_count + 1) * 1000:
            self.boss_spawn_count += 1
            new_boss = Boss(self.boss_spawn_count, self.player)
            self.objects.append(new_boss)
            bosses_alive.append(new_boss) 
            for obj in self.objects:
                if isinstance(obj, Enemy): obj.destroy()
            self.objects = [o for o in self.objects if o.is_alive or o is self.player]

        for obj in self.objects:
            if obj is not self.player: obj.update()

        current_boss_exists = len(bosses_alive) > 0
        if self.prev_boss_exists and not current_boss_exists and not self.player.is_dead:
            reward += 5000.0 
            
        self.prev_boss_exists = current_boss_exists

        px, py = self.player.x, self.player.y
        wall_dist_x = min(px, WIDTH - px)
        wall_dist_y = min(py, HEIGHT - py)
        min_wall_dist = min(wall_dist_x, wall_dist_y)

        if min_wall_dist < 100:
            reward -= (100 - min_wall_dist) * 0.02 

        if bosses_alive:
            nearest_boss = min(bosses_alive, key=lambda b: math.hypot(b.x - self.player.x, b.y - self.player.y))
            bx, by = nearest_boss.x, nearest_boss.y
            boss_dist = math.hypot(bx - px, by - py)
            
            DANGER_ZONE = 400
            
            if boss_dist < DANGER_ZONE:
                if 250 < boss_dist < 400:
                    reward += 3.0
                elif boss_dist < 150:
                    reward -= 5.0 
                    
                dx, dy = px - bx, py - by
                dist_norm = math.hypot(dx, dy) + 1e-5
                dir_x, dir_y = dx / dist_norm, dy / dist_norm
                
                vx, vy = self.player.vx, self.player.vy
                dot_product = (dir_x * vx) + (dir_y * vy)
                
                if dot_product > 0: reward += dot_product * 1.5 
                else: reward += dot_product * 2.0 
                
                speed_norm = math.hypot(vx, vy)
                if speed_norm < self.player.speed * 0.5: reward -= 3.0 
                
                margin = 80 
                in_corner = (px < margin or px > WIDTH - margin) and (py < margin or py > HEIGHT - margin)
                if in_corner: reward -= 15.0 
            
            self.prev_boss_dist = boss_dist
        else:
            self.prev_boss_dist = None

        if self.player.hp / self.player.max_hp <= 0.4:
            heals = [h for h in self.objects if isinstance(h, HealPack)]
            if heals:
                nearest_heal = min(heals, key=lambda h: math.hypot(h.x - self.player.x, h.y - self.player.y))
                hx, hy = nearest_heal.x - self.player.x, nearest_heal.y - self.player.y
                dist = math.hypot(hx, hy)
                if dist > 0:
                    dot = (self.player.vx * (hx/dist) + self.player.vy * (hy/dist))
                    if dot > 0: reward += 5.0
                    elif dot < 0: reward -= 5.0

        bullets = [o for o in self.objects if o.is_bullet]
        player_targets = [o for o in self.objects if o is not self.player and o.can_collide_with_player]
        bullet_targets = [o for o in self.objects if o is not self.player and o.can_collide_with_bullet]

        for obj in player_targets:
            if obj.is_alive and self.player.rect.colliderect(obj.rect): obj.on_player_collision(self.player)

        for bullet in bullets:
            for obj in bullet_targets:
                if bullet.is_alive and obj.is_alive and bullet.rect.colliderect(obj.rect):
                    obj.on_bullet_collision(bullet, self.player)

        self.objects = [o for o in self.objects if o.is_alive or o is self.player]

        if not current_boss_exists:
            enemy_exists = any(isinstance(o, Enemy) for o in self.objects)
            if not enemy_exists:
                for _ in range(6): self.objects.append(Enemy(self.player))

        hp_diff = self.player.hp - self.prev_hp
        if hp_diff < 0: reward += hp_diff * 10.0
        elif hp_diff > 0: reward += hp_diff * 5.0
            
        if self.player.is_dead:
            reward -= 500.0
            done = True

        self.prev_hp = self.player.hp
        score_diff = self.player.score - self.prev_score
        if score_diff > 0: reward += score_diff * 5.0 
        self.prev_score = self.player.score

        return self.get_state(), reward, done

def draw_ui(surface, player, record_100k, mode, speed, episode, is_training_paused):
    bar_x, bar_y = 20, 20
    bar_w, bar_h = 220, 24

    pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_w, bar_h))
    current_w = bar_w * (player.hp / player.max_hp)
    if current_w > 0: pygame.draw.rect(surface, GREEN, (bar_x, bar_y, current_w, bar_h))
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, bar_w, bar_h), 2)

    hp_text = font.render(f"HP: {int(player.hp)}/{int(player.max_hp)}", True, BLACK)
    score_text = font.render(f"Score: {player.score}", True, BLACK)
    
    if mode == 2:
        mode_str = "AI EXPLOIT (T: OFF)" if is_training_paused else "AI TRAIN (T: ON)"
        info_text1 = font.render("T: Toggle Train   S: Save   L: Load   [/]: Speed   R: Restart   ESC: Quit", True, BLACK)
    else:
        mode_str = "HUMAN"
        info_text1 = font.render("SPACE: Shoot   F: Dash(Lv.10)   R: Restart   ESC: Quit", True, BLACK)
        
    info_text2 = font.render(f"MODE: {mode_str} | SPEED: {speed}x | EP: {episode}", True, BLUE)

    surface.blit(hp_text, (20, 50))
    surface.blit(score_text, (20, 80))
    surface.blit(info_text1, (20, 110))
    surface.blit(info_text2, (20, 140))

    if record_100k:
        record_text = small_font.render(f"10만점 돌파: {record_100k}", True, BLUE)
        surface.blit(record_text, (WIDTH - record_text.get_width() - 20, 20))

def draw_level_up(surface, player):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))
    title = big_font.render("보스 처치! 보상을 클릭하세요", True, WHITE)
    surface.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

    card_w, card_h = 240, 300
    gap = 40
    start_x = (WIDTH - (card_w * 3 + gap * 2)) // 2
    y = 180
    cards = []
    
    next_atk_lvl = player.atk_lvl + 1
    next_atk_bonus = 50.0 / math.sqrt(next_atk_lvl)

    if player.attack_cooldown <= 100.0:
        extra_bullets = int(player.bonus_bullet_pct // 100)
        rem_pct = player.bonus_bullet_pct % 100
        current_spd_str = f"100ms (+{extra_bullets}발, {rem_pct:.1f}%)"
        next_add_pct = 25.0 * 2.0 / math.sqrt(player.spd_lvl + 1)
        effect_spd_str = f"게이지 +{next_add_pct:.1f}%"
    else:
        current_spd_str = f"{player.attack_cooldown:.1f}ms"
        effect_spd_str = "딜레이 -25%"

    texts = [
        (f"공격력 (Lv.{player.atk_lvl})", f"현재: {player.attack_power:.1f}", f"데미지 +{next_atk_bonus:.1f}%"),
        (f"공격속도 (Lv.{player.spd_lvl})", f"현재: {current_spd_str}", effect_spd_str),
        (f"이동속도 (Lv.{player.mov_lvl})", f"현재: {player.speed:.1f}", "속도 +0.5")
    ]
    perk_texts = ["Lv.10: 적 최대체력 비례 데미지", "Lv.10: [스페이스] 자동 연사", "Lv.10: [F] 대시 (AI 미지원)"]

    for i in range(3):
        rect = pygame.Rect(start_x + i * (card_w + gap), y, card_w, card_h)
        cards.append(rect)
        pygame.draw.rect(surface, WHITE, rect)
        pygame.draw.rect(surface, GOLD, rect, 5)

        title_text = font.render(texts[i][0], True, BLACK)
        desc_text = tiny_font.render(texts[i][1], True, GRAY)
        effect_text = font.render(texts[i][2], True, BLUE)
        lvl_check = [player.atk_lvl, player.spd_lvl, player.mov_lvl][i]
        perk_color = PURPLE if lvl_check >= 10 else GRAY
        perk_font = tiny_font.render(perk_texts[i], True, perk_color)

        surface.blit(title_text, (rect.centerx - title_text.get_width() // 2, rect.y + 40))
        surface.blit(desc_text, (rect.centerx - desc_text.get_width() // 2, rect.y + 100))
        surface.blit(effect_text, (rect.centerx - effect_text.get_width() // 2, rect.y + 160))
        surface.blit(perk_font, (rect.centerx - perk_font.get_width() // 2, rect.y + 240))
    return cards

def draw_game_over(surface, player, final_time_ms):
    msg1 = big_font.render("GAME OVER", True, RED)
    msg2 = font.render(f"Final Score: {player.score}", True, BLACK)
    time_str = format_time(final_time_ms)
    msg_time = font.render(f"Play Time: {time_str}", True, BLUE)
    msg3 = font.render("R: Restart   ESC: Quit", True, BLACK)

    surface.blit(msg1, (WIDTH // 2 - msg1.get_width() // 2, 200))
    surface.blit(msg2, (WIDTH // 2 - msg2.get_width() // 2, 260))
    surface.blit(msg_time, (WIDTH // 2 - msg_time.get_width() // 2, 300))
    surface.blit(msg3, (WIDTH // 2 - msg3.get_width() // 2, 350))

class MainApp:
    def __init__(self):
        self.env = GameEnv()
        self.agent = DQNAgent(state_dim=35, action_dim=9) 
        self.state = "MENU"
        self.mode = 1
        self.speed = 1
        self.episode = 0
        self.manual_action = None
        self.ui_msg = ""
        self.ui_msg_timer = 0
        self.cards_rects = []
        self.game_over_pause = False
        self.action_hold_timer = 0
        self.last_ai_action = 0
        self.is_training_paused = False
        self.speed_steps = [1, 5, 10, 20, 50, 100]

    def draw_menu(self):
        screen.fill(WHITE)
        title = big_font.render("SHOOTER AI - FULL FEATURES", True, BLUE)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 150))
        menus = ["[1] HUMAN PLAY", "[2] AI TRAIN"]
        for i, text in enumerate(menus):
            t = font.render(text, True, BLACK)
            screen.blit(t, (WIDTH // 2 - t.get_width() // 2, 250 + i * 50))
            
        if self.ui_msg_timer > 0:
            msg_surf = font.render(self.ui_msg, True, RED)
            screen.blit(msg_surf, (WIDTH // 2 - msg_surf.get_width() // 2, 500))
            self.ui_msg_timer -= 1

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                if self.state == "MENU":
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_1: 
                            self.mode, self.state = 1, "PLAYING"
                            self.env.reset(); self.speed = 1; self.game_over_pause = False
                        if event.key == pygame.K_2: 
                            self.mode, self.state = 2, "PLAYING"
                            self.env.reset(); self.speed = 1; self.game_over_pause = False
                            self.is_training_paused = False
                            if self.agent.epsilon < 0.1: self.agent.epsilon = 0.5 
                            self.action_hold_timer = 0; self.last_ai_action = 0

                elif self.state == "PLAYING":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.mode == 1 and self.env.player.pending_level_ups > 0:
                            for i, rect in enumerate(self.cards_rects):
                                if rect.collidepoint(event.pos): self.manual_action = 6 + i
                                
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE: self.state = "MENU"
                        if event.key == pygame.K_s and self.mode == 2: 
                            self.agent.save()
                            self.ui_msg = "Model Saved!"; self.ui_msg_timer = 60
                        if event.key == pygame.K_l and self.mode == 2: 
                            self.agent.load()
                            self.ui_msg = "Model Loaded!"; self.ui_msg_timer = 60
                            
                        if event.key == pygame.K_t and self.mode == 2:
                            self.is_training_paused = not self.is_training_paused
                            self.ui_msg = "Train OFF (Exploitation)" if self.is_training_paused else "Train ON (Exploration)"
                            self.ui_msg_timer = 90
                            
                        if event.key == pygame.K_RIGHTBRACKET:
                            idx = self.speed_steps.index(self.speed) if self.speed in self.speed_steps else 0
                            self.speed = self.speed_steps[min(idx + 1, len(self.speed_steps)-1)]
                        if event.key == pygame.K_LEFTBRACKET:
                            idx = self.speed_steps.index(self.speed) if self.speed in self.speed_steps else 0
                            self.speed = self.speed_steps[max(idx - 1, 0)]
                        
                        if self.game_over_pause and event.key == pygame.K_r:
                            self.env.reset()
                            self.game_over_pause = False
                            self.action_hold_timer = 0; self.last_ai_action = 0
                            
                        if self.mode == 1 and not self.game_over_pause and event.key == pygame.K_f:
                            self.env.player.dash(pygame.mouse.get_pos())

            if self.state == "MENU":
                self.draw_menu()
            
            elif self.state == "PLAYING" and not self.game_over_pause:
                for _ in range(self.speed):
                    obs = self.env.get_state()
                    action = None
                    is_training_active = (self.mode == 2 and not self.is_training_paused)

                    if self.mode == 1:
                        if self.env.player.pending_level_ups > 0:
                            if self.manual_action is not None:
                                action = self.manual_action
                                self.manual_action = None
                            else: break
                        else:
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_SPACE]: action = 5
                    else:
                        if self.env.player.pending_level_ups > 0:
                            action = self.agent.act(obs, env=self.env, last_action=self.last_ai_action, train=is_training_active, pending_level_up=True)
                            self.action_hold_timer = 0 
                        else:
                            if self.action_hold_timer <= 0:
                                action = self.agent.act(obs, env=self.env, last_action=self.last_ai_action, train=is_training_active, pending_level_up=False)
                                self.last_ai_action = action
                                self.action_hold_timer = 5 
                            else:
                                action = self.last_ai_action
                                self.action_hold_timer -= 1

                    next_obs, reward, done = self.env.step(action)

                    if is_training_active:
                        self.agent.memory.append((obs, action if action is not None else 0, reward, next_obs, done))
                        self.agent.train_step()

                    if done:
                        self.episode += 1
                        if self.mode == 2: 
                            if is_training_active:
                                self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * self.agent.epsilon_decay)
                            if self.episode % 5 == 0:
                                self.agent.save()
                            self.env.reset()
                            self.action_hold_timer = 0
                            self.last_ai_action = 0
                        else:
                            self.game_over_pause = True
                        break

            if self.state == "PLAYING" and (self.speed <= 20 or self.episode % 2 == 0 or self.game_over_pause):
                screen.fill(WHITE)
                for obj in self.env.objects: obj.draw(screen)
                
                draw_ui(screen, self.env.player, self.env.record_100k_time, self.mode, self.speed, self.episode, self.is_training_paused)
                
                if self.ui_msg_timer > 0:
                    msg_surf = font.render(self.ui_msg, True, RED)
                    screen.blit(msg_surf, (WIDTH // 2 - msg_surf.get_width() // 2, 400))
                    self.ui_msg_timer -= 1
                
                if self.env.player.pending_level_ups > 0:
                    self.cards_rects = draw_level_up(screen, self.env.player)
                    
                if self.game_over_pause:
                    draw_game_over(screen, self.env.player, self.env.play_time_ms)
            
            pygame.display.flip()
            if self.speed <= 5: clock.tick(60)

if __name__ == "__main__":
    MainApp().run()