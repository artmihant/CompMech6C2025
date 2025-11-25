import { Vector2 } from '../utils/math.js';
import { CONFIG } from '../config.js';

export class Enemy {
    constructor(pos, vel, targetPlanet) {
        this.r = pos;
        this.v = vel;
        this.targetPlanet = targetPlanet; // Ссылка на объект планеты
        this.radius = CONFIG.ENEMY.radius;
        this.color = CONFIG.ENEMY.color;
        this.markedForDeath = false;
    }
    
    get state() { return { r: this.r, v: this.v }; }
    set state(s) { this.r = s.r; this.v = s.v; }
}

export class Projectile {
    constructor(pos, vel) {
        this.r = pos;
        this.v = vel;
        this.radius = CONFIG.PROJECTILE.radius;
        this.color = CONFIG.PROJECTILE.color;
        this.ttl = CONFIG.PROJECTILE.ttl;
        this.markedForDeath = false;
    }

    get state() { return { r: this.r, v: this.v }; }
    set state(s) { this.r = s.r; this.v = s.v; }
}

