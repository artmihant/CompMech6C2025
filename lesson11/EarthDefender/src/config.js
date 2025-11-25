// Игровые параметры и константы

export const CONFIG = {
    // Физика и мир
    G: 2.5, // Увеличено для ощутимой гравитации
    SCALE: 1.0,  // Масштаб (GU -> Pixels), пока 1:1
    DT_SIM: 0.04, // Шаг симуляции (с)
    WORLD_RADIUS: 700, // Граница мира ("Пояс астероидов")

    // Геймплей
    PLANET_HP: 4, // Попаданий до заражения
    INFECTED_SPAWN_INTERVAL: 5.0, // Секунды между спавном из зараженной планеты

    // Солнце
    SUN: {
        mass: 100000,
        radius: 40,
        color: '#FFFF00'
    },

    // Земля
    EARTH: {
        mass: 1000,
        orbitRadius: 300,
        orbitPeriod: 60, // секунды
        radius: 12,
        color: '#0000FF'
    },

    // Луна
    MOON: {
        mass: 10,
        orbitRadius: 40,
        orbitPeriod: 6,
        radius: 4,
        color: '#AAAAAA'
    },

    // Другие планеты (примерные значения для баланса)
    MERCURY: {
        mass: 300,
        orbitRadius: 120,
        orbitPeriod: 20,
        radius: 6,
        color: '#CC7722'
    },
    VENUS: {
        mass: 800,
        orbitRadius: 200,
        orbitPeriod: 40,
        radius: 10,
        color: '#EEDD82'
    },
    MARS: {
        mass: 600,
        orbitRadius: 450,
        orbitPeriod: 110,
        radius: 9,
        color: '#B22222'
    },

    // Игрок
    PLAYER: {
        radius: 8,
        color: '#00FF00',
        thrust: 2, // Сила тяги
        rotationSpeed: 1.0, // Рад/с
        fireRate: 0.1, // Выстрелов в секунду
        fireRange: 600 // Примерная зона для отрисовки
    },

    // Снаряд
    PROJECTILE: {
        speedRel: 200, // Скорость относительно корабля
        radius: 2,
        ttl: 8.0, // Время жизни (с)
        color: '#FFFFFF'
    },

    // Враги
    ENEMY: {
        radius: 5,
        color: '#FF0000',
        spawnRadius: 700, // Спавн на границе мира
        waveInterval: 20, // Секунды между волнами
        baseWaveSize: 6,
        launchSpeed: 40 // Начальная скорость запуска в сторону планеты
    }
};
