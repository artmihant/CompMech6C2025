import pygame
import math

# --- КОНФИГУРАЦИЯ ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
HALF_WIDTH = SCREEN_WIDTH // 2
HALF_HEIGHT = SCREEN_HEIGHT // 2
FPS = 60
TILE_SIZE = 50
FOV = math.pi / 3
HALF_FOV = FOV / 2
NUM_RAYS = SCREEN_WIDTH // 2  # Уменьшаем кол-во лучей для оптимизации (иначе будет 2 FPS)
MAX_DEPTH = 800
DELTA_ANGLE = FOV / NUM_RAYS
DIST = NUM_RAYS / (2 * math.tan(HALF_FOV))
PROJ_COEFF = 3 * DIST * TILE_SIZE
SCALE = SCREEN_WIDTH // NUM_RAYS

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 0, 0)
GREEN = (0, 220, 0)
BLUE = (0, 0, 220)
DARKGRAY = (40, 40, 40)
SKY_BLUE = (135, 206, 235)
FLOOR_COLOR = (30, 30, 30)

# Карта (текстовый массив)
text_map = [
    '111111111111',
    '100000000001',
    '100001000001',
    '100001000001',
    '100222220001',
    '100200020001',
    '100000000001',
    '111111111111'
]

# Парсинг карты
world_map = {}
for j, row in enumerate(text_map):
    for i, char in enumerate(row):
        if char != '0':
            world_map[(i * TILE_SIZE, j * TILE_SIZE)] = char

# --- ИГРОК ---
class Player:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.angle = 0
        self.speed = 2
        self.rot_speed = 0.02

    def movement(self):
        keys = pygame.key.get_pressed()
        sin_a = math.sin(self.angle)
        cos_a = math.cos(self.angle)
        dx = 0
        dy = 0
        if keys[pygame.K_w]:
            dx += self.speed * cos_a
            dy += self.speed * sin_a
        if keys[pygame.K_s]:
            dx += -self.speed * cos_a
            dy += -self.speed * sin_a
        if keys[pygame.K_a]:
            dx += self.speed * sin_a
            dy += -self.speed * cos_a
        if keys[pygame.K_d]:
            dx += -self.speed * sin_a
            dy += self.speed * cos_a
        
        if keys[pygame.K_LEFT]:
            self.angle -= self.rot_speed
        if keys[pygame.K_RIGHT]:
            self.angle += self.rot_speed

        # Простая проверка коллизий (не дает войти в стену)
        self.check_collision(dx, dy)

    def check_collision(self, dx, dy):
        # Проверяем будущую позицию
        if not self.is_wall(self.x + dx, self.y):
            self.x += dx
        if not self.is_wall(self.x, self.y + dy):
            self.y += dy

    def is_wall(self, x, y):
        # Вычисляем координаты клетки
        grid_x = int(x // TILE_SIZE)
        grid_y = int(y // TILE_SIZE)
        # Проверка границ массива
        if 0 <= grid_y < len(text_map) and 0 <= grid_x < len(text_map[0]):
            return text_map[grid_y][grid_x] != '0'
        return False

# --- RAYCASTING ---
def ray_casting(sc, player_pos, player_angle):
    cur_angle = player_angle - HALF_FOV
    xo, yo = player_pos
    for ray in range(NUM_RAYS):
        sin_a = math.sin(cur_angle)
        cos_a = math.cos(cur_angle)

        # Наивный алгоритм: идем мелкими шагами пока не врежемся
        # Это ОЧЕНЬ медленно, но просто реализуется
        for depth in range(1, MAX_DEPTH):
            x = xo + depth * cos_a
            y = yo + depth * sin_a
            
            # Проверка на стену
            # Приводим к индексам сетки
            grid_x = int(x // TILE_SIZE)
            grid_y = int(y // TILE_SIZE)
            
            if 0 <= grid_y < len(text_map) and 0 <= grid_x < len(text_map[0]):
                char = text_map[grid_y][grid_x]
                if char != '0':
                    # Нашли стену!
                    # Убираем эффект рыбьего глаза
                    depth *= math.cos(player_angle - cur_angle)
                    
                    # Высота стены
                    proj_height = PROJ_COEFF / (depth + 0.0001)
                    
                    # Выбор цвета (просто по символу)
                    c = WHITE
                    if char == '1': c = GREEN
                    elif char == '2': c = RED
                    
                    # Затемнение от дальности (простейшее)
                    color = [max(0, min(255, int(Comp / (1 + depth * 0.005)))) for Comp in c]
                    
                    # Рисуем полосу
                    pygame.draw.rect(sc, color, 
                                     (ray * SCALE, HALF_HEIGHT - proj_height // 2, SCALE, proj_height))
                    break
            else:
                break # Вышли за карту

        cur_angle += DELTA_ANGLE

# --- MAIN ---
pygame.init()
sc = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
player = Player()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    player.movement()

    # Отрисовка
    # Потолок и пол
    pygame.draw.rect(sc, SKY_BLUE, (0, 0, SCREEN_WIDTH, HALF_HEIGHT))
    pygame.draw.rect(sc, FLOOR_COLOR, (0, HALF_HEIGHT, SCREEN_WIDTH, HALF_HEIGHT))

    # 3D вид
    ray_casting(sc, (player.x, player.y), player.angle)

    # Миникарта
    # Рисуем клетки
    for y, row in enumerate(text_map):
        for x, char in enumerate(row):
            if char != '0':
                pygame.draw.rect(sc, DARKGRAY, (x * TILE_SIZE // 4, y * TILE_SIZE // 4, TILE_SIZE // 4, TILE_SIZE // 4))
    # Рисуем игрока на миникарте
    pygame.draw.circle(sc, RED, (int(player.x // 4), int(player.y // 4)), 3)

    pygame.display.flip()
    clock.tick(FPS)
    print(f"FPS: {clock.get_fps():.1f}")

