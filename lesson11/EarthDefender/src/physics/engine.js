import { Vector2, rk4_step } from '../utils/math.js';
import { CONFIG } from '../config.js';
import { Enemy, Projectile } from '../entities/Other.js';

/**
 * Вычисляет ускорение в точке r для объекта (не влияющего своей массой на другие)
 * @param {Vector2} r - позиция объекта
 * @param {Array<Planet>} bodies - массив массивных тел (планет, солнца)
 * @returns {Vector2} Вектор ускорения
 */
export function calculateGravity(r, bodies) {
    let a = new Vector2(0, 0);
    
    const rSun = new Vector2(0, 0);
    const diffSun = rSun.sub(r);
    const distSqSun = diffSun.magSq();
    const distSun = Math.sqrt(distSqSun);
    
    if (distSun > CONFIG.SUN.radius) { 
         const forceMag = CONFIG.G * CONFIG.SUN.mass / distSqSun; 
         a = a.add(diffSun.norm().mul(forceMag));
    }

    for (const body of bodies) {
        const diff = body.pos.sub(r);
        const distSq = diff.magSq();
        const dist = Math.sqrt(distSq);
        
        if (dist > body.radius) {
            const forceMag = CONFIG.G * body.mass / distSq;
            a = a.add(diff.norm().mul(forceMag));
        }
    }
    
    return a;
}

export class PhysicsEngine {
    constructor() {
        this.waveTimer = 0;
        this.infectedSpawnTimer = 0;
    }

    step(world, dt) {
        if (world.gameOver) return;

        // 1. Обновляем массивные тела (кинематика)
        world.time += dt;
        for (const planet of world.planets) {
            planet.update(world.time);
        }
        
        // Проверка заражения Земли (Game Over)
        const earth = world.planets.find(p => p.mass === CONFIG.EARTH.mass && p.orbitRadius === CONFIG.EARTH.orbitRadius);
        if (earth && earth.isInfected) {
             world.gameOver = true;
             return;
        }

        const gravityFn = (r, v) => calculateGravity(r, world.planets);

        // 2. Игрок
        const shipAccelFn = (r, v) => {
            let a = gravityFn(r, v);
            if (world.ship.thrusting) {
                const thrustDir = Vector2.fromAngle(world.ship.angle);
                a = a.add(thrustDir.mul(CONFIG.PLAYER.thrust)); 
            }
            return a;
        };

        const newShipState = rk4_step(world.ship.state, dt, shipAccelFn);
        world.ship.state = newShipState;
        
        // Проверка выхода игрока за границы мира -> Респавн в L4
        if (world.ship.r.mag() > CONFIG.WORLD_RADIUS && earth) {
             // L4 находится под углом +60 градусов от Земли
             const earthAngle = Math.atan2(earth.pos.y, earth.pos.x);
             const l4Angle = earthAngle + Math.PI / 3;
             
             const l4Pos = new Vector2(Math.cos(l4Angle), Math.sin(l4Angle)).mul(earth.orbitRadius);
             
             // Вектор скорости L4
             const l4Vel = new Vector2(-Math.sin(l4Angle), Math.cos(l4Angle)).mul(earth.omega * earth.orbitRadius);
             
             world.ship.respawn(l4Pos, l4Vel);
        }
        
        if (world.ship.cooldownTimer > 0) {
            world.ship.cooldownTimer -= dt;
        }

        // 3. Враги
        for (let i = world.enemies.length - 1; i >= 0; i--) {
            const enemy = world.enemies[i];
            const newState = rk4_step(enemy.state, dt, gravityFn);
            enemy.state = newState;

            // Коллизия с планетами
            for (const planet of world.planets) {
                if (enemy.r.dist(planet.pos) < planet.radius + enemy.radius) {
                    planet.takeHit();
                    enemy.markedForDeath = true;
                }
            }
            
            if (enemy.r.mag() < CONFIG.SUN.radius + enemy.radius) {
                 enemy.markedForDeath = true;
            }

            if (enemy.r.mag() > CONFIG.WORLD_RADIUS + 50) {
                enemy.markedForDeath = true;
            }
            
            if (enemy.markedForDeath) {
                world.enemies.splice(i, 1);
            }
        }

        // 4. Снаряды
        for (let i = world.projectiles.length - 1; i >= 0; i--) {
            const proj = world.projectiles[i];
            proj.ttl -= dt;
            if (proj.ttl <= 0) {
                world.projectiles.splice(i, 1);
                continue;
            }
            
            if (proj.r.mag() > CONFIG.WORLD_RADIUS) {
                world.projectiles.splice(i, 1);
                continue;
            }

            const newState = rk4_step(proj.state, dt, gravityFn);
            proj.state = newState;
            
            for (const enemy of world.enemies) {
                if (!enemy.markedForDeath && proj.r.dist(enemy.r) < proj.radius + enemy.radius) {
                    enemy.markedForDeath = true;
                    proj.markedForDeath = true; 
                }
            }
            
            if (proj.markedForDeath) {
                world.projectiles.splice(i, 1);
            }
        }
        
        // 5. Спавн
        this.handleSpawning(world, dt);
        this.handleInfectedSpawns(world, dt);
        
        // 6. Авто-стрельба
        this.handleAutoFire(world);
    }

    handleSpawning(world, dt) {
        this.waveTimer += dt;
        if (this.waveTimer > CONFIG.ENEMY.waveInterval) {
            this.waveTimer = 0;
            this.spawnWave(world);
        }
    }

    spawnWave(world) {
        const count = CONFIG.ENEMY.baseWaveSize;
        const launchSpeed = CONFIG.ENEMY.launchSpeed;

        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const startPos = Vector2.fromAngle(angle, CONFIG.ENEMY.spawnRadius);
            const target = world.planets[Math.floor(Math.random() * world.planets.length)];
            
            this.spawnSingleEnemy(world, startPos, target, launchSpeed);
        }
    }
    
    handleInfectedSpawns(world, dt) {
        this.infectedSpawnTimer += dt;
        if (this.infectedSpawnTimer > CONFIG.INFECTED_SPAWN_INTERVAL) {
            this.infectedSpawnTimer = 0;
            
            const infectedPlanets = world.planets.filter(p => p.isInfected);
            for (const planet of infectedPlanets) {
                // Ищем цель (не зараженную планету)
                const targets = world.planets.filter(p => p !== planet && !p.isInfected);
                if (targets.length === 0) continue;
                
                const target = targets[Math.floor(Math.random() * targets.length)];
                
                // Спавним чуть выше атмосферы
                const startPos = planet.pos.add(new Vector2(Math.random()-0.5, Math.random()-0.5).norm().mul(planet.radius + 10));
                
                // Начальная скорость врага = скорость планеты + launchSpeed
                // Но spawnSingleEnemy рассчитывает скорость запуска.
                // Нам нужно адаптировать логику.
                // У зараженных планет запуск должен учитывать, что враг уже имеет скорость планеты.
                
                // Модифицированная версия: передаем baseVelocity
                this.spawnSingleEnemy(world, startPos, target, CONFIG.ENEMY.launchSpeed, planet.vel);
            }
        }
    }

    spawnSingleEnemy(world, startPos, target, launchSpeed, baseVelocity = new Vector2(0,0)) {
        // Итеративный расчет упреждения
        let timeToHit = startPos.dist(target.pos) / launchSpeed;
        
        for(let k=0; k<3; k++) {
            const futurePos = target.getPosAtTime(world.time + timeToHit);
            const dist = startPos.dist(futurePos);
            timeToHit = dist / launchSpeed;
        }
        
        const predictedPos = target.getPosAtTime(world.time + timeToHit);
        const dirToTarget = predictedPos.sub(startPos).norm();
        
        // Если мы стартуем с планеты, у нас уже есть её скорость (baseVelocity).
        // Если из космоса - baseVelocity = 0.
        // Мы хотим, чтобы RELATIVE speed была launchSpeed.
        // V_total = V_base + Dir * LaunchSpeed
        
        const startVel = baseVelocity.add(dirToTarget.mul(launchSpeed));
        
        world.enemies.push(new Enemy(startPos, startVel, target));
    }

    handleAutoFire(world) {
        if (world.ship.cooldownTimer > 0) return;

        let bestTarget = null;
        let minDist = CONFIG.PLAYER.fireRange;
        let bestAimDir = null;

        const projSpeed = CONFIG.PROJECTILE.speedRel;
        const shipVel = world.ship.v;

        for (const enemy of world.enemies) {
            const dist = world.ship.r.dist(enemy.r);
            if (dist > minDist) continue;

            const P = enemy.r.sub(world.ship.r);
            const V = enemy.v.sub(shipVel);
            const S = projSpeed;

            const A = V.magSq() - S*S;
            const B = 2 * (P.x * V.x + P.y * V.y);
            const C = P.magSq();

            let t = -1;
            if (Math.abs(A) < 1e-6) {
                t = -C / B;
            } else {
                const disc = B*B - 4*A*C;
                if (disc >= 0) {
                    const sqrtDisc = Math.sqrt(disc);
                    const t1 = (-B - sqrtDisc) / (2*A);
                    const t2 = (-B + sqrtDisc) / (2*A);
                    if (t1 > 0 && t2 > 0) t = Math.min(t1, t2);
                    else if (t1 > 0) t = t1;
                    else if (t2 > 0) t = t2;
                }
            }

            if (t > 0 && t < CONFIG.PROJECTILE.ttl) {
                const interceptPos = enemy.r.add(enemy.v.mul(t));
                
                if (dist < minDist) { 
                    minDist = dist;
                    bestTarget = enemy;
                    const vBullet = interceptPos.sub(world.ship.r).div(t);
                    bestAimDir = vBullet.sub(shipVel).norm();
                }
            }
        }

        if (bestTarget && bestAimDir) {
            const projVel = world.ship.v.add(bestAimDir.mul(projSpeed));
            world.projectiles.push(new Projectile(world.ship.r.copy(), projVel));
            world.ship.cooldownTimer = 1.0 / CONFIG.PLAYER.fireRate;
        }
    }
}
