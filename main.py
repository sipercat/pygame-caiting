import pygame
import random
import sys
import math
import asyncio
from abc import ABC, abstractmethod

pygame.init()
WIDTH, HEIGHT = 900, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Encapsulation & Level Up System")
clock = pygame.time.Clock()

font = pygame.font.SysFont("malgungothic", 20)
small_font = pygame.font.SysFont("malgungothic", 15)
tiny_font = pygame.font.SysFont("malgungothic", 10)
big_font = pygame.font.SysFont("malgungothic", 40)

WHITE, BLACK, BLUE, RED, GOLD, GREEN, GRAY, PURPLE = (245, 245, 245), (30, 30, 30), (70, 130, 255), (220, 80, 80), (255,
                                                                                                                    210,
                                                                                                                    0), (
    80, 200, 120), (180, 180, 180), (148, 0, 211)


# 밀리초 단위의 누적 시간을 '분:초.밀리초' 형태의 직관적인 문자열로 변환해주는 함수
def format_time(ms_time):
    seconds = (ms_time // 1000) % 60
    minutes = ms_time // 60000
    millis = (ms_time % 1000) // 10
    return f"{minutes:02d}:{seconds:02d}.{millis:02d}"


# 게임 내 화면에 존재하는 모든 객체(플레이어, 적, 아이템 등)의 부모, 추상 클래스
class GameObject(ABC):
    def __init__(self):
        self._alive = True

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


# 체력을 가지고 피해를 입을 수 있는 생명체(플레이어, 적, 보스)들의 부모 클래스
class LivingEntity(GameObject):
    def __init__(self, x, y, radius, max_hp):
        super().__init__()
        self._x = x
        self._y = y
        self._radius = radius
        self._max_hp = float(max_hp)
        self._hp = self._max_hp
        self._debuffs = []

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def radius(self):
        return self._radius

    @property
    def max_hp(self):
        return self._max_hp

    @property
    def hp(self):
        return self._hp

    @hp.setter
    def hp(self, value):
        self._hp = max(0.0, min(self._max_hp, float(value)))

    @property
    def rect(self):
        return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)

    def take_damage(self, amount, killer=None):
        if not self.is_alive: return
        self.hp -= amount
        if self.hp <= 0:
            self.die(killer)

    def die(self, killer=None):
        self.destroy()

    def apply_debuff(self, dps, duration, killer):
        self._debuffs.append({"dps": dps, "time": duration, "killer": killer})

    def update_debuffs(self, dt):
        alive_debuffs = []
        for db in self._debuffs:
            self.take_damage(db["dps"] * dt, db["killer"])
            db["time"] -= dt
            if db["time"] > 0 and self.is_alive:
                alive_debuffs.append(db)
        self._debuffs = alive_debuffs

    def draw_hp_bar(self, surface, y_offset=-8):
        if self._max_hp > 1 or isinstance(self, Enemy):
            bar_w = self._radius * 2
            bar_h = 5
            bar_x = int(self._x - self._radius)
            bar_y = int(self._y - self._radius + y_offset)
            pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_w, bar_h))
            current_w = bar_w * (self._hp / self._max_hp)
            if current_w > 0:
                bar_color = GREEN if isinstance(self, Enemy) else RED
                pygame.draw.rect(surface, bar_color, (bar_x, bar_y, current_w, bar_h))
            pygame.draw.rect(surface, BLACK, (bar_x, bar_y, bar_w, bar_h), 1)


# 유저가 조종하며 성장(레벨업)과 스킬(대시, 연사)을 사용하는, 캐릭터에 관한 기능을 담은 클래스
class Player(LivingEntity):
    def __init__(self):
        super().__init__(x=100, y=300, radius=20, max_hp=100)
        self._speed = 2.5
        self._attack_power = 5.0
        self._score = 0

        # [핵심 시스템: 사격 쿨타임과 경직]
        # __attack_cooldown: 한 번 쏘고 다음 총알이 나갈 때까지 걸리는 딜레이 (기본 0.4초)
        # __next_shoot_time: 현재 시간(ticks)과 비교하여 총을 쏠 수 있는지 판별하는 기준점
        # __freeze_end_time: 총을 쏜 직후 발생하는 반동(경직)으로 인해 이동이 불가능한 시간
        self.__freeze_end_time = 0
        self.__attack_cooldown = 400.0
        self.__next_shoot_time = 0

        self.pending_level_ups = 0
        self.atk_lvl = 0
        self.spd_lvl = 0
        self.mov_lvl = 0

        self.dash_ready = False
        self.bonus_bullet_pct = 0.0

    @property
    def score(self):
        return self._score

    @property
    def attack_power(self):
        return self._attack_power

    @property
    def attack_cooldown(self):
        return self.__attack_cooldown

    @property
    def speed(self):
        return self._speed

    def add_score(self, amount):
        if amount >= 0: self._score += amount

    def refill_dash(self):
        if self.mov_lvl >= 10:
            self.dash_ready = True

    # [Lv.10 특성: 이동속도]
    # 이동속도가 10레벨이 되면 해금되는 기능으로, 마우스 커서 방향으로 순간 이동
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

    # 최대체력 비례 추가 데미지
    def upgrade_attack_power(self):
        self.atk_lvl += 1
        bonus_pct = 50.0 / math.sqrt(self.atk_lvl)
        self._attack_power *= (1.0 + (bonus_pct / 100.0))

    def upgrade_attack_speed(self):
        self.spd_lvl += 1
        target_cooldown = self.__attack_cooldown * 0.75
        # 쿨타임 100ms를 기점으로 더는 못 줄이고 남은 쿨타임 %비율로 추가 탄환 생성
        if target_cooldown < 100.0:
            if self.__attack_cooldown > 100.0:
                applied_pct = ((self.__attack_cooldown - 100.0) / self.__attack_cooldown) * 100.0
                excess_pct = 25.0 - applied_pct
                self.__attack_cooldown = 100.0
            else:
                excess_pct = 25.0

            if excess_pct > 0:
                added_pct = excess_pct * 2.0 / math.sqrt(self.spd_lvl)
                self.bonus_bullet_pct += added_pct
        else:
            self.__attack_cooldown = target_cooldown
    # 레벨업 중 유일하게 등차함수로 증가하는 능력
    def upgrade_move_speed(self):
        self.mov_lvl += 1
        self._speed += 0.5
        if self.mov_lvl >= 10:
            self.dash_ready = True

    def shoot(self, objects_ref, is_auto=False):
        current_time = pygame.time.get_ticks()

        # 쿨타임 확인 로직
        if current_time < self.__next_shoot_time:
            return []

        # [Lv.10 특성: 공격속도]
        # is_auto(자동 사격:키다운) 발동 시 쿨타임을 강제로 200ms로 고정하여 밸런스 조절
        effective_cooldown = 200.0 if is_auto else self.__attack_cooldown

        # 다음 발사 가능 시간 갱신
        self.__next_shoot_time = current_time + effective_cooldown

        # 사격 직후 이동 불가 시간(경직) 적용 (쿨타임의 8분의 3)
        # 이 기능이 이 게임의 핵심. 때리면 본인도 무조건 멈춰야 해서 거리 조절을 잘 해야 함, 이를 통해 얻는 재미를 타겟으로 함
        freeze_duration = effective_cooldown * 0.375
        self.__freeze_end_time = current_time + freeze_duration

        bullets = []
        total_bullets = 1 + int(self.bonus_bullet_pct // 100)
        gap = 12
        start_y = self._y - (total_bullets - 1) * gap / 2.0

        for i in range(total_bullets):
            b_y = start_y + i * gap
            bullets.append(Bullet(self._x + self._radius, b_y, objects_ref, self))
        return bullets

    def update(self):
        if not self.is_alive or pygame.time.get_ticks() < self.__freeze_end_time:
            return

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  self._x -= self._speed
        if keys[pygame.K_RIGHT]: self._x += self._speed
        if keys[pygame.K_UP]:    self._y -= self._speed
        if keys[pygame.K_DOWN]:  self._y += self._speed

        self._x = max(self._radius, min(WIDTH - self._radius, self._x))
        self._y = max(self._radius, min(HEIGHT - self._radius, self._y))

    def draw(self, surface):
        color = GRAY if not self.is_alive else BLUE
        pygame.draw.circle(surface, color, (int(self._x), int(self._y)), self._radius)

        if self.mov_lvl >= 10 and self.dash_ready:
            pygame.draw.circle(surface, GOLD, (int(self._x), int(self._y)), 6)


# 화면 우측에서 스폰되어 왼쪽으로 돌진하기만 하는 적. 처치 시 점수를 준다
class Enemy(LivingEntity):
    def __init__(self, player):
        super().__init__(x=0, y=0, radius=20, max_hp=2)
        self.player = player
        self._speed = random.choice([3, 4, 5, 6])
        self.reset()

    @property
    def can_collide_with_player(self):
        return True

    @property
    def can_collide_with_bullet(self):
        return True

    def reset(self):
        self._alive = True
        self._x = WIDTH + self._radius
        self._y = random.randint(50, HEIGHT - 50)
        scaling_hp = 2 + (self.player.score // 200)
        self._max_hp = float(scaling_hp)
        self.hp = self._max_hp
        self._debuffs.clear()

    def die(self, killer=None):
        if killer:
            killer.add_score(200)
        self.reset()

    def update(self):
        self.update_debuffs(1 / 60.0)
        self._x -= self._speed
        if self._x < -self._radius:
            self.reset()

    def draw(self, surface):
        pygame.draw.circle(surface, RED, (int(self._x), int(self._y)), self._radius)
        self.draw_hp_bar(surface)

    def on_player_collision(self, player):
        player.take_damage(1)

    def on_bullet_collision(self, bullet, player):
        if self.is_alive and bullet.is_alive:
            self.take_damage(player.attack_power, player)

            # [Lv.10 특성: 공격력]
            # 공격력 10레벨 도달 시, 적의 '최대 체력'에 비례하는 추가 데미지를 넣는다
            if player.atk_lvl >= 10:
                bonus_dmg = self.max_hp * (math.sqrt(math.log(player.atk_lvl, 3)) / 100.0)
                self.take_damage(bonus_dmg, player)

            bullet.destroy()


# 특정 점수 구간(1000점)마다 등장하며 플레이어를 추적하고 처치 시 레벨업 기회를 주는 보스. 등장할 때마다 체력 및 공격력이 sqrt(n)에 비례하여 증가한다.
class Boss(LivingEntity):
    def __init__(self, spawn_count, player):
        max_hp = int(75 * math.sqrt(spawn_count))
        super().__init__(x=0, y=0, radius=40, max_hp=max_hp)
        self.player = player
        self._speed = 1.1 * math.sqrt(spawn_count)

        while True:
            self._x = random.randint(self._radius, WIDTH - self._radius)
            self._y = random.randint(self._radius, HEIGHT - self._radius)
            dist = math.hypot(self._x - self.player.x, self._y - self.player.y)
            if dist >= 300:
                break

    @property
    def can_collide_with_player(self):
        return True

    @property
    def can_collide_with_bullet(self):
        return True

    def die(self, killer=None):
        if killer:
            killer.add_score(200)
            killer.pending_level_ups += 1
            killer.refill_dash()
        super().die(killer)

    def update(self):
        if not self.is_alive: return
        self.update_debuffs(1 / 60.0)
        dx = self.player.x - self._x
        dy = self.player.y - self._y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self._x += (dx / dist) * self._speed
            self._y += (dy / dist) * self._speed

    def draw(self, surface):
        pygame.draw.circle(surface, PURPLE, (int(self._x), int(self._y)), self._radius)
        self.draw_hp_bar(surface, y_offset=5)

    def on_player_collision(self, player):
        player.take_damage(2)

    def on_bullet_collision(self, bullet, player):
        if self.is_alive and bullet.is_alive:
            self.take_damage(player.attack_power, player)
            if player.atk_lvl >= 10:
                bonus_dmg = self.max_hp * (math.sqrt(math.log(player.atk_lvl, 3)) / 100.0)
                self.take_damage(bonus_dmg, player)
            bullet.destroy()


# 플레이어가 획득하면 점수를 올려주는 수집형 재화 아이템
class Coin(GameObject):
    def __init__(self):
        super().__init__()
        self._radius = 12
        self._dy = 1
        self._direction = 1
        self.reset()

    @property
    def can_collide_with_player(self): return True

    @property
    def rect(self):
        return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)

    def reset(self):
        self._x = random.randint(100, WIDTH - 50)
        self._y = random.randint(50, HEIGHT - 50)

    def update(self):
        self._y += self._dy * self._direction
        if self._y > HEIGHT - 30 or self._y < 30:
            self._direction *= -1

    def draw(self, surface):
        pygame.draw.circle(surface, GOLD, (int(self._x), int(self._y)), self._radius)
        pygame.draw.circle(surface, BLACK, (int(self._x), int(self._y)), self._radius, 2)

    def on_player_collision(self, player):
        player.add_score(10)
        self.reset()


# 플레이어가 획득하면 잃은 체력을 즉시 회복시켜주는 아이템
class HealPack(GameObject):
    def __init__(self):
        super().__init__()
        self._radius = 15
        self._heal_amount = 20
        self.reset()

    @property
    def can_collide_with_player(self): return True

    @property
    def rect(self):
        return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)

    def reset(self):
        self._x = random.randint(100, WIDTH - 50)
        self._y = random.randint(50, HEIGHT - 50)

    def update(self): pass

    def draw(self, surface):
        pygame.draw.circle(surface, GREEN, (int(self._x), int(self._y)), self._radius)
        pygame.draw.line(surface, WHITE, (self._x, self._y - 8), (self._x, self._y + 8), 4)
        pygame.draw.line(surface, WHITE, (self._x - 8, self._y), (self._x + 8, self._y), 4)

    def on_player_collision(self, player):
        player.hp += self._heal_amount
        self.reset()


# 플레이어가 발사하여 화면 내의 적을 향해 유도 비행(호밍 시스템)하며 데미지를 입힌다. 레벨업에 따라 공격력이 달라진다
class Bullet(GameObject):
    def __init__(self, x, y, objects_ref, player_ref):
        super().__init__()
        self._x = x
        self._y = y
        self._speed = 10
        self._radius = 5
        self._objects = objects_ref
        self._player = player_ref
        self._dx = self._speed
        self._dy = 0

    @property
    def is_bullet(self):
        return True

    @property
    def rect(self):
        return pygame.Rect(int(self._x - self._radius), int(self._y - self._radius), self._radius * 2, self._radius * 2)

    def update(self):
        target = None
        min_dist = float('inf')

        for obj in self._objects:
            if obj.is_alive and (isinstance(obj, Enemy) or isinstance(obj, Boss)):
                dist = math.hypot(obj.rect.centerx - self._x, obj.rect.centery - self._y)
                if dist < min_dist:
                    min_dist = dist
                    target = obj

        if target and min_dist > 0:
            self._dx = ((target.rect.centerx - self._x) / min_dist) * self._speed
            self._dy = ((target.rect.centery - self._y) / min_dist) * self._speed

        self._x += self._dx
        self._y += self._dy

        if self._x < -10 or self._x > WIDTH + 10 or self._y < -10 or self._y > HEIGHT + 10:
            self.destroy()

    def draw(self, surface):
        pygame.draw.circle(surface, BLACK, (int(self._x), int(self._y)), self._radius)


# 플레이어의 현재 체력, 점수, 조작법, 10만점 달성 시간 등 게임 플레이 정보를 화면 상단에 그리는 UI
def draw_ui(surface, player, record_100k):
    bar_x, bar_y = 20, 20
    bar_w, bar_h = 220, 24

    pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_w, bar_h))
    current_w = bar_w * (player.hp / player.max_hp)
    pygame.draw.rect(surface, GREEN, (bar_x, bar_y, current_w, bar_h))
    pygame.draw.rect(surface, BLACK, (bar_x, bar_y, bar_w, bar_h), 2)

    hp_text = font.render(f"HP: {int(player.hp)}/{int(player.max_hp)}", True, BLACK)
    score_text = font.render(f"Score: {player.score}", True, BLACK)
    info_text = font.render("SPACE: Shoot   F: Dash(Lv.10)   R: Restart   ESC: Quit", True, BLACK)

    surface.blit(hp_text, (20, 50))
    surface.blit(score_text, (20, 80))
    surface.blit(info_text, (20, 110))

    if record_100k:
        record_text = small_font.render(f"10만점 돌파: {record_100k}", True, BLUE)
        surface.blit(record_text, (WIDTH - record_text.get_width() - 20, 20))


# 플레이어가 사망했을 때 화면 중앙에 최종 획득 점수와 누적 플레이 타임을 띄워주는 게임 오버 화면
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


# 보스를 처치한 후 게임을 일시정지시키고 3가지(공격력/공격속도/이동속도) 선택지를 제공하는 렙업 창
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

    perk_texts = [
        "Lv.10: 적 최대체력 비례 추가 데미지",
        "Lv.10: [스페이스] 자동 연사 (200ms)",
        "Lv.10: [F] 마우스 방향 대시"
    ]

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


# 게임을 최초로 시작하거나 사망 후 재시작할 때 플레이어와 적, 아이템들을 초기화함
def create_objects():
    player = Player()
    objects = [player]
    for _ in range(6): objects.append(Enemy(player))
    for _ in range(10): objects.append(Coin())
    for _ in range(2): objects.append(HealPack())
    return player, objects


player, objects = create_objects()
boss_spawn_count = 0

card_w, card_h = 240, 300
gap = 40
start_x = (WIDTH - (card_w * 3 + gap * 2)) // 2
card_rects = [pygame.Rect(start_x + i * (card_w + gap), 180, card_w, card_h) for i in range(3)]

play_time_ms = 0
last_tick = pygame.time.get_ticks()
record_100k_time = None
async def main():
    global player, objects
    global boss_spawn_count
    global play_time_ms
    global last_tick
    global record_100k_time

    while True:
        clock.tick(60)

        current_tick = pygame.time.get_ticks()
        dt_ms = current_tick - last_tick
        last_tick = current_tick

        if player.is_alive and player.pending_level_ups == 0:
            play_time_ms += dt_ms

            if player.score >= 100000 and record_100k_time is None:
                record_100k_time = format_time(play_time_ms)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if player.pending_level_ups > 0:
                    mx, my = event.pos

                    for i, rect in enumerate(card_rects):
                        if rect.collidepoint(mx, my):

                            if i == 0:
                                player.upgrade_attack_power()

                            elif i == 1:
                                player.upgrade_attack_speed()

                            elif i == 2:
                                player.upgrade_move_speed()

                            player.pending_level_ups -= 1
                            break
            if event.type == pygame.QUIT:
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

                if player.pending_level_ups == 0:
                    if not player.is_dead and event.key == pygame.K_SPACE:
                        bullets = player.shoot(objects)
                        if bullets:
                            objects.extend(bullets)

                    if not player.is_dead and event.key == pygame.K_f:
                        player.dash(pygame.mouse.get_pos())

                    if player.is_dead and event.key == pygame.K_r:
                        player, objects = create_objects()
                        boss_spawn_count = 0
                        play_time_ms = 0
                        record_100k_time = None

        if not player.is_dead and player.pending_level_ups == 0:

            keys = pygame.key.get_pressed()

            # 공격속도 Lv10 자동연사
            if keys[pygame.K_SPACE] and player.spd_lvl >= 10:
                bullets = player.shoot(objects, is_auto=True)
                if bullets:
                    objects.extend(bullets)

            # 보스 생성
            if player.score >= (boss_spawn_count + 1) * 1000:
                boss_spawn_count += 1
                objects.append(Boss(boss_spawn_count, player))

            # 객체 업데이트
            for obj in objects:
                obj.update()

            # 충돌 판정
            bullets = [obj for obj in objects if obj.is_bullet]

            player_targets = [
                obj for obj in objects
                if obj is not player and obj.can_collide_with_player
            ]

            bullet_targets = [
                obj for obj in objects
                if obj is not player and obj.can_collide_with_bullet
            ]

            for obj in player_targets:
                if obj.is_alive and player.rect.colliderect(obj.rect):
                    obj.on_player_collision(player)

            for bullet in bullets:
                for obj in bullet_targets:
                    if (bullet.is_alive and
                            obj.is_alive and
                            bullet.rect.colliderect(obj.rect)):
                        obj.on_bullet_collision(bullet, player)

            objects = [obj for obj in objects if obj.is_alive or obj is player]

        screen.fill(WHITE)

        for obj in objects:
            obj.draw(screen)

        draw_ui(screen, player, record_100k_time)

        if player.pending_level_ups > 0:
            draw_level_up(screen, player)

        elif player.is_dead:
            draw_game_over(screen, player, play_time_ms)

        pygame.display.flip()

        await asyncio.sleep(0)


asyncio.run(main())
