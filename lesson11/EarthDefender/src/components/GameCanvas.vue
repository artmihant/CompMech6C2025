<template>
    <div class="game-container">
        <canvas ref="canvas"></canvas>
        <div class="ui-overlay">
            <div>Time: {{ time.toFixed(1) }}</div>
            <div>Enemies: {{ enemyCount }}</div>
            <div>Infected: {{ infectedCount }}</div>
            <div class="controls-hint">Controls: A/D to Rotate, SPACE to Thrust</div>
        </div>
        <div v-if="gameOver" class="game-over">
            <h1>GAME OVER</h1>
            <p>Earth has been infected!</p>
            <button @click="restart">Try Again</button>
        </div>
    </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import { CONFIG } from '../config.js';
import { Vector2 } from '../utils/math.js';
import { Planet } from '../entities/Planet.js';
import { Ship } from '../entities/Ship.js';
import { PhysicsEngine } from '../physics/engine.js';

const canvas = ref(null);
const time = ref(0);
const enemyCount = ref(0);
const infectedCount = ref(0);
const gameOver = ref(false);

// World state
let world = {
    time: 0,
    planets: [],
    ship: null,
    enemies: [],
    projectiles: [],
    gameOver: false
};

const engine = new PhysicsEngine();
let ctx = null;
let animationFrameId = null;
let lastTime = 0;

// Input state
const keys = {
    ArrowLeft: false,
    ArrowRight: false,
    a: false,
    d: false,
    Space: false,
    " ": false
};

function initWorld() {
    gameOver.value = false;
    world = {
        time: 0,
        planets: [],
        ship: null,
        enemies: [],
        projectiles: [],
        gameOver: false
    };

    // Инициализация планет
    const earth = new Planet(CONFIG.EARTH);
    const moon = new Planet(CONFIG.MOON, 0, earth.pos); 
    
    world.planets = [
        earth,
        moon,
        new Planet(CONFIG.MERCURY, Math.random() * 6),
        new Planet(CONFIG.VENUS, Math.random() * 6),
        new Planet(CONFIG.MARS, Math.random() * 6)
    ];
    
    world.earthRef = earth;
    world.moonRef = moon;

    world.ship = new Ship();
    // Начальная позиция игрока в L4
    const earthAngle = Math.atan2(earth.pos.y, earth.pos.x);
    const l4Angle = earthAngle + Math.PI / 3;
    const l4Pos = new Vector2(Math.cos(l4Angle), Math.sin(l4Angle)).mul(earth.orbitRadius);
    const l4Vel = new Vector2(-Math.sin(l4Angle), Math.cos(l4Angle)).mul(earth.omega * earth.orbitRadius);
    world.ship.respawn(l4Pos, l4Vel);
}

function restart() {
    initWorld();
    lastTime = performance.now();
    // Если loop был остановлен (хотя мы его не останавливаем полностью, просто физика может встать)
    // Но здесь физика в engine.step проверяет gameOver.
}

function handleInput() {
    if (gameOver.value) return;
    const left = keys.ArrowLeft || keys.a;
    const right = keys.ArrowRight || keys.d;
    const thrust = keys.Space || keys[" "];
    
    const dt = CONFIG.DT_SIM; 
    
    if (left) world.ship.angle -= CONFIG.PLAYER.rotationSpeed * dt;
    if (right) world.ship.angle += CONFIG.PLAYER.rotationSpeed * dt;
    
    world.ship.thrusting = thrust;
}

function updateMoonCenter() {
    world.moonRef.center = world.earthRef.pos;
}

function gameLoop(timestamp) {
    const dt = (timestamp - lastTime) / 1000;
    lastTime = timestamp;
    
    const safeDt = Math.min(dt, 0.1);
    
    handleInput();

    if (!world.gameOver) {
        updateMoonCenter();
        engine.step(world, CONFIG.DT_SIM);
    } else {
        gameOver.value = true;
    }

    time.value = world.time;
    enemyCount.value = world.enemies.length;
    infectedCount.value = world.planets.filter(p => p.isInfected).length;

    draw();
    
    animationFrameId = requestAnimationFrame(gameLoop);
}

function draw() {
    if (!ctx) return;
    const width = canvas.value.width;
    const height = canvas.value.height;
    
    // Clear
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);
    
    // Camera transform
    const minDim = Math.min(width, height);
    const scale = minDim / (CONFIG.WORLD_RADIUS * 2 * 1.05); 
    
    ctx.save();
    ctx.translate(width / 2, height / 2);
    ctx.scale(scale, scale);
    
    // "Пояс астероидов"
    ctx.beginPath();
    ctx.arc(0, 0, CONFIG.WORLD_RADIUS + 5000, 0, Math.PI * 2); 
    ctx.arc(0, 0, CONFIG.WORLD_RADIUS, 0, Math.PI * 2, true); 
    ctx.fillStyle = '#111111'; 
    ctx.fill();
    
    // Граница мира
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 2 / scale;
    ctx.beginPath();
    ctx.arc(0, 0, CONFIG.WORLD_RADIUS, 0, Math.PI * 2);
    ctx.stroke();

    // Орбиты
    ctx.lineWidth = 1 / scale; 
    world.planets.forEach(p => {
        ctx.strokeStyle = '#333';
        ctx.beginPath();
        ctx.arc(p.center.x, p.center.y, p.orbitRadius, 0, Math.PI * 2);
        ctx.stroke();
    });

    // Солнце
    ctx.fillStyle = CONFIG.SUN.color;
    ctx.beginPath();
    ctx.arc(0, 0, CONFIG.SUN.radius, 0, Math.PI * 2);
    ctx.fill();
    
    // Планеты
    world.planets.forEach(p => {
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.pos.x, p.pos.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // HP
        if (!p.isInfected) {
             const hp = CONFIG.PLANET_HP - p.hits;
             // Рисуем всегда или только если < MAX? ТЗ просит обратный отсчет. Рисуем всегда для наглядности.
             ctx.fillStyle = '#FFF';
             ctx.font = `bold ${14/scale}px monospace`; 
             ctx.textAlign = 'center';
             ctx.textBaseline = 'middle';
             ctx.fillText(hp, p.pos.x, p.pos.y);
        } else {
             // Infected
             ctx.fillStyle = '#000'; // Черный восклицательный знак на фиолетовой планете
             ctx.font = `bold ${14/scale}px monospace`;
             ctx.textAlign = 'center';
             ctx.textBaseline = 'middle';
             ctx.fillText('!', p.pos.x, p.pos.y);
        }
    });

    // Игрок
    if (!world.gameOver) {
        const ship = world.ship;
        ctx.save();
        ctx.translate(ship.r.x, ship.r.y);
        ctx.rotate(ship.angle);
        
        ctx.fillStyle = ship.color;
        ctx.beginPath();
        ctx.moveTo(10, 0);
        ctx.lineTo(-7, 7);
        ctx.lineTo(-7, -7);
        ctx.fill();
        
        if (ship.thrusting) {
            ctx.fillStyle = '#FFA500';
            ctx.beginPath();
            ctx.moveTo(-7, 0);
            ctx.lineTo(-12, 3);
            ctx.lineTo(-15, 0);
            ctx.lineTo(-12, -3);
            ctx.fill();
        }
        ctx.restore();
        
        // Зона стрельбы
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.1)';
        ctx.lineWidth = 1 / scale;
        ctx.beginPath();
        ctx.arc(ship.r.x, ship.r.y, CONFIG.PLAYER.fireRange, 0, Math.PI * 2);
        ctx.stroke();
    }

    // Враги
    world.enemies.forEach(e => {
        ctx.fillStyle = e.color;
        ctx.beginPath();
        ctx.arc(e.r.x, e.r.y, e.radius, 0, Math.PI * 2);
        ctx.fill();
    });

    // Снаряды
    world.projectiles.forEach(p => {
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.r.x, p.r.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
    });

    ctx.restore();
}

function resize() {
    if (canvas.value) {
        canvas.value.width = window.innerWidth;
        canvas.value.height = window.innerHeight;
    }
}

onMounted(() => {
    ctx = canvas.value.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
    
    window.addEventListener('keydown', (e) => keys[e.key] = true);
    window.addEventListener('keyup', (e) => keys[e.key] = false);
    
    initWorld();
    lastTime = performance.now();
    animationFrameId = requestAnimationFrame(gameLoop);
});

onUnmounted(() => {
    window.removeEventListener('resize', resize);
    window.removeEventListener('keydown', (e) => keys[e.key] = true);
    window.removeEventListener('keyup', (e) => keys[e.key] = false);
    cancelAnimationFrame(animationFrameId);
});
</script>

<style scoped>
.game-container {
    position: relative;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
}
canvas {
    display: block;
}
.ui-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    color: white;
    font-family: monospace;
    background: rgba(0, 0, 0, 0.5);
    padding: 10px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 10;
}
.controls-hint {
    margin-top: 10px;
    font-size: 0.8em;
    opacity: 0.8;
}
.game-over {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(50, 0, 0, 0.9);
    color: white;
    padding: 40px;
    text-align: center;
    border-radius: 10px;
    border: 2px solid red;
    z-index: 20;
}
.game-over h1 {
    margin: 0 0 20px;
    color: red;
}
.game-over button {
    background: #333;
    color: white;
    border: 1px solid white;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 1.2em;
}
.game-over button:hover {
    background: #555;
}
</style>
