import { Vector2 } from '../utils/math.js';
import { CONFIG } from '../config.js';

export class Ship {
    constructor() {
        // Начальная позиция будет перезаписана при initWorld, 
        // но зададим безопасный дефолт
        this.r = new Vector2(CONFIG.EARTH.orbitRadius, 0); 
        this.v = new Vector2(0, 0); 
        
        this.angle = -Math.PI / 2; // Смотрит вверх
        this.thrusting = false;
        
        this.radius = CONFIG.PLAYER.radius;
        this.color = CONFIG.PLAYER.color;
        
        // Стрельба
        this.cooldownTimer = 0;
    }

    respawn(pos, vel) {
        if (pos) this.r = pos.copy();
        else this.r = new Vector2(CONFIG.EARTH.orbitRadius * 0.5, 0); // Fallback

        if (vel) this.v = vel.copy();
        else this.v = new Vector2(0, 0);

        this.angle = -Math.PI / 2;
        this.thrusting = false;
    }

    get state() {
        return { r: this.r, v: this.v };
    }
    
    set state(s) {
        this.r = s.r;
        this.v = s.v;
    }
}
