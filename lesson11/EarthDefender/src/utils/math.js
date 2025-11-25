export class Vector2 {
    constructor(x, y) {
        this.x = x || 0;
        this.y = y || 0;
    }

    add(v) {
        return new Vector2(this.x + v.x, this.y + v.y);
    }

    sub(v) {
        return new Vector2(this.x - v.x, this.y - v.y);
    }

    mul(scalar) {
        return new Vector2(this.x * scalar, this.y * scalar);
    }

    div(scalar) {
        if (scalar === 0) return new Vector2(0, 0);
        return new Vector2(this.x / scalar, this.y / scalar);
    }

    mag() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    magSq() {
        return this.x * this.x + this.y * this.y;
    }

    norm() {
        const m = this.mag();
        if (m === 0) return new Vector2(0, 0);
        return this.div(m);
    }

    dist(v) {
        return this.sub(v).mag();
    }
    
    copy() {
        return new Vector2(this.x, this.y);
    }

    static fromAngle(angle, length = 1) {
        return new Vector2(Math.cos(angle) * length, Math.sin(angle) * length);
    }
}

/**
 * Классический RK4 интегратор.
 * @param {Object} state - Текущее состояние { r: Vector2, v: Vector2 }
 * @param {number} dt - Шаг времени
 * @param {Function} accelerationFn - Функция (r, v) => Vector2 (возвращает ускорение)
 * @returns {Object} Новое состояние { r, v }
 */
export function rk4_step(state, dt, accelerationFn) {
    // k1
    const k1_v = accelerationFn(state.r, state.v);
    const k1_r = state.v;

    // k2
    const k2_r_state = state.r.add(k1_r.mul(dt / 2));
    const k2_v_state = state.v.add(k1_v.mul(dt / 2));
    const k2_v = accelerationFn(k2_r_state, k2_v_state);
    const k2_r = k2_v_state; // Внимание: в RK4 dx/dt = v. Здесь v берется из промежуточного шага.
                             // В ТЗ: k2_r = state.v + k1_v*(dt/2). Это в точности k2_v_state.

    // k3
    const k3_r_state = state.r.add(k2_r.mul(dt / 2));
    const k3_v_state = state.v.add(k2_v.mul(dt / 2));
    const k3_v = accelerationFn(k3_r_state, k3_v_state);
    const k3_r = k3_v_state;

    // k4
    const k4_r_state = state.r.add(k3_r.mul(dt));
    const k4_v_state = state.v.add(k3_v.mul(dt));
    const k4_v = accelerationFn(k4_r_state, k4_v_state);
    const k4_r = k4_v_state;

    // Итоговое суммирование
    // new_r = state.r + (dt/6)*(k1_r + 2*k2_r + 2*k3_r + k4_r)
    const dr = k1_r.add(k2_r.mul(2)).add(k3_r.mul(2)).add(k4_r).mul(dt / 6);
    const new_r = state.r.add(dr);

    // new_v = state.v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
    const dv = k1_v.add(k2_v.mul(2)).add(k3_v.mul(2)).add(k4_v).mul(dt / 6);
    const new_v = state.v.add(dv);

    return { r: new_r, v: new_v };
}

