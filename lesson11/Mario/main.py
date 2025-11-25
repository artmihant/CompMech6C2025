import pygame
import sys
import os
import random

# Инициализация Pygame
pygame.init()
pygame.font.init()

# --- Константы и Настройки ---
WIDTH, HEIGHT = 800, 600
FPS = 60
TITLE = "Emoji Mario"
HIGHSCORE_FILE = "highscore.txt"

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (135, 206, 235) # Небо
RED = (255, 0, 0)
GOLD = (255, 215, 0)
GROUND_COLOR = (101, 67, 33) # Коричневый
BLOCK_COLOR = (139, 69, 19) # Темно-коричневый

# Спрайты (Эмодзи)
SPRITE_HERO = "😎"
SPRITE_HERO_JUMP = "😲"
SPRITE_HERO_DEAD = "😵"
SPRITE_HERO_WIN = "🤩"
SPRITE_BLOCK = "🧱"
SPRITE_ENEMY = "👾"
SPRITE_OBSTACLE = "🔥"
SPRITE_GROUND = "🟩"
SPRITE_COIN = "🪙"

# Физика
PLAYER_SPEED = 5
JUMP_FORCE = 15
GRAVITY = 0.8

# Пути к шрифтам (попытка найти системные шрифты с эмодзи/символами)
FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/System/Library/Fonts/Apple Color Emoji.ttc", 
]

# --- Классы ---

class EmojiSprite(pygame.sprite.Sprite):
    def __init__(self, x, y, emoji, size=40, is_ground=False):
        super().__init__()
        self.emoji = emoji
        self.size = size
        self.is_ground = is_ground
        self.original_y = y
        self.pos_x = float(x)
        self.pos_y = float(y)
        self.image = self.render_emoji(emoji, size, is_ground)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def render_emoji(self, emoji, size, is_ground):
        # Создаем поверхность фиксированного размера с поддержкой прозрачности
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Если это блок или земля, заливаем фон
        if is_ground:
             pygame.draw.rect(surf, GROUND_COLOR, (0, 0, size, size))
        elif emoji == SPRITE_BLOCK:
             pygame.draw.rect(surf, BLOCK_COLOR, (0, 0, size, size))
             pygame.draw.rect(surf, BLACK, (0, 0, size, size), 1) # Контур

        # Находим шрифт
        font = None
        for path in FONT_PATHS:
            if os.path.exists(path):
                try:
                    font = pygame.font.Font(path, int(size * 0.8)) 
                    break
                except:
                    continue
        if font is None:
             sys_fonts = ['segoeuiemoji', 'notocoloremoji', 'applecoloremoji', 'arial', 'dejavusans', 'freesansbold']
             for f_name in sys_fonts:
                try:
                    font = pygame.font.SysFont(f_name, int(size * 0.8))
                    break
                except:
                    continue
        if font is None:
             font = pygame.font.SysFont(None, int(size * 0.8))

        try:
            text_color = BLACK 
            if is_ground: text_color = (34, 139, 34)
            if emoji == SPRITE_HERO: text_color = (255, 215, 0)
            if emoji == SPRITE_HERO_JUMP: text_color = (255, 140, 0)
            if emoji == SPRITE_HERO_DEAD: text_color = (100, 100, 100)
            if emoji == SPRITE_OBSTACLE: text_color = RED
            if emoji == SPRITE_ENEMY: text_color = (128, 0, 128)
            if emoji == SPRITE_COIN: text_color = GOLD

            text_surface = font.render(emoji, True, text_color)
            if text_surface.get_width() == 0: raise Exception("Empty")

            text_rect = text_surface.get_rect(center=(size // 2, size // 2))
            surf.blit(text_surface, text_rect)
            
        except Exception:
            if not is_ground and emoji != SPRITE_BLOCK: 
                color = RED if emoji == SPRITE_ENEMY else (255, 255, 0)
                if emoji in [SPRITE_HERO, SPRITE_HERO_JUMP, SPRITE_HERO_DEAD]: color = (255, 200, 0)
                if emoji == SPRITE_COIN: color = GOLD
                pygame.draw.circle(surf, color, (size // 2, size // 2), size // 2 - 4)

        return surf
    
    def set_emoji(self, emoji):
        if self.emoji != emoji:
            self.emoji = emoji
            self.image = self.render_emoji(emoji, self.size, self.is_ground)

class Enemy(EmojiSprite):
    def __init__(self, x, y, blocks):
        super().__init__(x, y, SPRITE_ENEMY, 40, is_ground=False)
        self.blocks = blocks
        self.vel_y = 0
        self.on_ground = False
        # Прыгаем на месте, сила выше чем у игрока (15)
        self.jump_force = 22 
        
    def update(self):
        # Гравитация
        self.vel_y += GRAVITY
        self.pos_y += self.vel_y
        self.rect.y = round(self.pos_y)
        
        self.on_ground = False
        self.check_collision_y()
        
        # Если на земле - прыгаем
        if self.on_ground:
            self.vel_y = -self.jump_force
            self.on_ground = False

    def check_collision_y(self):
        hits = pygame.sprite.spritecollide(self, self.blocks, False)
        for block in hits:
            if self.vel_y > 0: # Падаем вниз
                self.rect.bottom = block.rect.top
                self.vel_y = 0
                self.on_ground = True
            elif self.vel_y < 0: # Прыгаем вверх
                self.rect.top = block.rect.bottom
                self.vel_y = 0
            self.pos_y = float(self.rect.y)

class Coin(EmojiSprite):
    def __init__(self, x, y):
        super().__init__(x, y, SPRITE_COIN, 30, is_ground=False)

class Player(EmojiSprite):
    def __init__(self, x, y):
        super().__init__(x, y, SPRITE_HERO)
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.start_pos = (x, y)
        self.score = 0

    def update(self, keys, blocks, enemies, obstacles, coins):
        # Движение влево/вправо
        self.vel_x = 0
        if keys[pygame.K_LEFT]:
            self.vel_x = -PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            self.vel_x = PLAYER_SPEED

        # Эмоции
        current_emoji = SPRITE_HERO
        
        # Прыжок
        if keys[pygame.K_SPACE] and self.on_ground:
            self.vel_y = -JUMP_FORCE
            self.on_ground = False
        
        if not self.on_ground:
             current_emoji = SPRITE_HERO_JUMP

        # Гравитация
        self.vel_y += GRAVITY

        # Применяем движение по X
        self.pos_x += self.vel_x
        self.rect.x = round(self.pos_x)
        self.check_collision_x(blocks)

        # Применяем движение по Y
        self.pos_y += self.vel_y
        self.rect.y = round(self.pos_y)
        self.on_ground = False 
        self.check_collision_y(blocks)
        
        self.set_emoji(current_emoji)

        # Сбор монеток
        coin_hits = pygame.sprite.spritecollide(self, coins, True) # True - удалять монетку
        for coin in coin_hits:
            self.score += 1

        # Проверка границ экрана (упал в пропасть)
        if self.rect.top > HEIGHT:
            self.set_emoji(SPRITE_HERO_DEAD)
            return "died"
        
        # Проверка столкновений с препятствиями (огонь - сразу смерть)
        if pygame.sprite.spritecollideany(self, obstacles):
            self.set_emoji(SPRITE_HERO_DEAD)
            return "died"
        
        # Проверка столкновений с врагами (Stomp механика)
        enemy_hits = pygame.sprite.spritecollide(self, enemies, False)
        for enemy in enemy_hits:
            # Если падаем сверху на врага
            if self.vel_y > 0 and self.rect.bottom < enemy.rect.bottom:
                enemy.kill() # Убить врага
                self.vel_y = -JUMP_FORCE * 0.5 # Отпрыгнуть
                self.pos_y = float(self.rect.y) # Обновляем pos_y для прыжка
                self.score += 5 # Бонус за убийство
            else:
                self.set_emoji(SPRITE_HERO_DEAD)
                return "died"
        
        return "alive"

    def check_collision_x(self, blocks):
        hits = pygame.sprite.spritecollide(self, blocks, False)
        for block in hits:
            if self.vel_x > 0: # Движемся вправо
                self.rect.right = block.rect.left
            elif self.vel_x < 0: # Движемся влево
                self.rect.left = block.rect.right
            # Важно: обновляем pos_x, чтобы не накапливалась ошибка
            self.pos_x = float(self.rect.x)

    def check_collision_y(self, blocks):
        hits = pygame.sprite.spritecollide(self, blocks, False)
        for block in hits:
            if self.vel_y > 0: # Падаем вниз
                self.rect.bottom = block.rect.top
                self.vel_y = 0
                self.on_ground = True
            elif self.vel_y < 0: # Прыгаем вверх и бьемся головой
                self.rect.top = block.rect.bottom
                self.vel_y = 0
            # Важно: обновляем pos_y
            self.pos_y = float(self.rect.y)

    def reset(self):
        self.rect.topleft = self.start_pos
        self.pos_x, self.pos_y = self.start_pos
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = "menu"
        self.font = pygame.font.SysFont("arial", 24)
        self.title_font = pygame.font.SysFont("arial", 48)
        
        self.all_sprites = pygame.sprite.Group()
        self.blocks = pygame.sprite.Group()
        self.enemies = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.coins = pygame.sprite.Group()
        
        self.level = 1
        self.total_score = 0 # Общий счет за сессию
        self.high_score = self.load_high_score()

        self.create_level()

    def load_high_score(self):
        if os.path.exists(HIGHSCORE_FILE):
            try:
                with open(HIGHSCORE_FILE, "r") as f:
                    return int(f.read())
            except:
                return 1
        return 1

    def save_high_score(self):
        if self.level > self.high_score:
            self.high_score = self.level
            try:
                with open(HIGHSCORE_FILE, "w") as f:
                    f.write(str(self.high_score))
            except:
                pass

    def create_level(self):
        self.all_sprites.empty()
        self.blocks.empty()
        self.enemies.empty()
        self.obstacles.empty()
        self.coins.empty()

        # Пол
        for i in range(0, WIDTH + 40, 40):
            b = EmojiSprite(i, HEIGHT - 40, SPRITE_GROUND, 40, is_ground=True)
            self.all_sprites.add(b)
            self.blocks.add(b)

        if self.level == 1:
            level_design = [
                (200, HEIGHT - 150, SPRITE_BLOCK),
                (240, HEIGHT - 150, SPRITE_BLOCK),
                (280, HEIGHT - 150, SPRITE_BLOCK),
                (400, HEIGHT - 250, SPRITE_BLOCK),
                (500, HEIGHT - 150, SPRITE_ENEMY),
                (600, HEIGHT - 80, SPRITE_OBSTACLE),
                (650, HEIGHT - 200, SPRITE_BLOCK),
            ]

            for x, y, kind in level_design:
                if kind == SPRITE_BLOCK:
                     s = EmojiSprite(x, y, kind, is_ground=False)
                     self.blocks.add(s)
                     self.all_sprites.add(s)
                elif kind == SPRITE_ENEMY:
                     s = Enemy(x, y, self.blocks)
                     self.enemies.add(s)
                     self.all_sprites.add(s)
                elif kind == SPRITE_OBSTACLE:
                     s = EmojiSprite(x, y, kind, is_ground=False)
                     self.obstacles.add(s)
                     self.all_sprites.add(s)
            
            # Монетки для первого уровня
            for cx in [220, 300, 420, 520]:
                 c = Coin(cx, HEIGHT - 300)
                 self.coins.add(c)
                 self.all_sprites.add(c)

        else:
            # Процедурная генерация
            num_platforms = random.randint(4, 7)
            for _ in range(num_platforms):
                x = random.randint(150, WIDTH - 150)
                y = random.randint(HEIGHT - 450, HEIGHT - 100)
                
                # Платформа
                s = EmojiSprite(x, y, SPRITE_BLOCK, is_ground=False)
                self.blocks.add(s)
                self.all_sprites.add(s)

                # Шанс спавна чего-либо на блоке
                roll = random.random()
                if roll < 0.2: # Враг
                    e = Enemy(x, y - 40, self.blocks)
                    self.enemies.add(e)
                    self.all_sprites.add(e)
                elif roll < 0.5: # Монетка
                    c = Coin(x + 5, y - 40)
                    self.coins.add(c)
                    self.all_sprites.add(c)
                elif roll < 0.6: # Препятствие
                     o = EmojiSprite(x, y - 40, SPRITE_OBSTACLE, is_ground=False)
                     self.obstacles.add(o)
                     self.all_sprites.add(o)
            
            # Наземные объекты
            for _ in range(random.randint(1, 3)):
                x = random.randint(200, WIDTH - 100)
                e = EmojiSprite(x, HEIGHT - 80, SPRITE_OBSTACLE, is_ground=False)
                self.obstacles.add(e)
                self.all_sprites.add(e)

        self.player = Player(50, HEIGHT - 100)
        self.player.score = self.total_score 
        self.all_sprites.add(self.player)

    def draw_text(self, text, font, color, x, y, align="topleft"):
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if align == "topleft":
            rect.topleft = (x, y)
        elif align == "center":
            rect.center = (x, y)
        elif align == "topright":
            rect.topright = (x, y)
        self.screen.blit(surface, rect)
        return rect

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()
        pygame.quit()
        sys.exit()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                
                if self.state == "menu":
                    if WIDTH // 2 - 50 <= mx <= WIDTH // 2 + 50 and HEIGHT // 2 - 20 <= my <= HEIGHT // 2 + 20:
                        self.state = "playing"
                        self.level = 1
                        self.total_score = 0
                        self.create_level()
                    if WIDTH // 2 - 50 <= mx <= WIDTH // 2 + 50 and HEIGHT // 2 + 40 <= my <= HEIGHT // 2 + 80:
                        self.state = "info"
                
                elif self.state == "info":
                    self.state = "menu"

                elif self.state == "playing" or self.state == "paused":
                    if WIDTH - 60 <= mx <= WIDTH - 10 and 10 <= my <= 50:
                        if self.state == "playing":
                            self.state = "paused"
                        else:
                            self.state = "playing"

    def update(self):
        if self.state == "playing":
            keys = pygame.key.get_pressed()
            self.enemies.update()
            status = self.player.update(keys, self.blocks, self.enemies, self.obstacles, self.coins)
            self.total_score = self.player.score

            if self.player.rect.left >= WIDTH - 20: 
                self.level += 1
                if self.level > self.high_score:
                    self.high_score = self.level
                    self.save_high_score()
                self.create_level()

            if status == "died":
                self.level = 1 
                self.total_score = 0
                self.create_level()

    def draw(self):
        self.screen.fill(BLUE)

        if self.state == "menu":
            self.screen.fill(WHITE)
            self.draw_text("MARIO EMOJI", self.title_font, BLACK, WIDTH // 2, HEIGHT // 4, "center")
            
            start_rect = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 - 20, 100, 40)
            pygame.draw.rect(self.screen, GRAY, start_rect)
            self.draw_text("START", self.font, WHITE, WIDTH // 2, HEIGHT // 2, "center")

            info_rect = pygame.Rect(WIDTH // 2 - 50, HEIGHT // 2 + 40, 100, 40)
            pygame.draw.rect(self.screen, GRAY, info_rect)
            self.draw_text("INFO", self.font, WHITE, WIDTH // 2, HEIGHT // 2 + 60, "center")

        elif self.state == "info":
            self.screen.fill(WHITE)
            self.draw_text("INFO", self.title_font, BLACK, WIDTH // 2, 50, "center")
            self.draw_text("Arrows: Move", self.font, BLACK, WIDTH // 2, 120, "center")
            self.draw_text("Space: Jump", self.font, BLACK, WIDTH // 2, 160, "center")
            self.draw_text("Jump on 👾 to kill", self.font, (128,0,128), WIDTH // 2, 200, "center")
            self.draw_text("Collect 🪙", self.font, GOLD, WIDTH // 2, 240, "center")
            self.draw_text("Avoid 🔥", self.font, RED, WIDTH // 2, 280, "center")
            
            self.draw_text(f"Max Level: {self.high_score}", self.font, BLUE, WIDTH // 2, 350, "center")
            self.draw_text("Click to back", self.font, GRAY, WIDTH // 2, 450, "center")

        elif self.state == "playing" or self.state == "paused":
            self.all_sprites.draw(self.screen)
            self.draw_text(f"Level: {self.level}", self.font, BLACK, 10, 10, "topleft")
            self.draw_text(f"Score: {self.total_score}", self.font, BLACK, 10, 40, "topleft")

            pygame.draw.rect(self.screen, WHITE, (WIDTH - 60, 10, 50, 40))
            pygame.draw.rect(self.screen, BLACK, (WIDTH - 60, 10, 50, 40), 2)
            display_text = "||" if self.state == "playing" else ">"
            self.draw_text(display_text, self.font, BLACK, WIDTH - 35, 30, "center")

            if self.state == "paused":
                s = pygame.Surface((WIDTH, HEIGHT))
                s.set_alpha(128)
                s.fill(BLACK)
                self.screen.blit(s, (0,0))
                self.draw_text("PAUSED", self.title_font, WHITE, WIDTH // 2, HEIGHT // 2, "center")

        pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()
