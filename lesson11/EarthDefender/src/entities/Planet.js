import { Vector2 } from '../utils/math.js';
import { CONFIG } from '../config.js';

export class Planet {
    constructor(config, angleOffset = 0, center = new Vector2(0, 0)) {
        this.mass = config.mass;
        this.radius = config.radius;
        this.color = config.color;
        this.orbitRadius = config.orbitRadius;
        this.orbitPeriod = config.orbitPeriod;
        this.center = center;

        // Вычисляем угловую скорость (рад/с)
        this.omega = (2 * Math.PI) / this.orbitPeriod;
        this.phase = angleOffset;

        this.pos = new Vector2(0, 0);
        this.vel = new Vector2(0, 0);

        // Состояние заражения
        this.hits = 0;
        this.isInfected = false;

        this.update(0);
    }

    getPosAtTime(t) {
        const angle = this.omega * t + this.phase;
        const rVec = new Vector2(Math.cos(angle) * this.orbitRadius, Math.sin(angle) * this.orbitRadius);
        return this.center.add(rVec);
    }

    update(t) {
        this.pos = this.getPosAtTime(t);

        const angle = this.omega * t + this.phase;
        this.vel = new Vector2(-Math.sin(angle), Math.cos(angle)).mul(this.omega * this.orbitRadius);
    }

    takeHit() {
        this.hits++;
        if (!this.isInfected && this.hits >= CONFIG.PLANET_HP) {
            this.isInfected = true;
            this.color = '#552255';
        }
    }
}
