from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Tuple
from queue import Queue, Empty

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from interface import (
    Float,
    Index,
    SolverInterface,
    SolverTime,
    NodesGeometryInput,
    NodesStateInput,
    PrimeElementTypesInput,
    CommonElementsInput,
    MaterialsInput,
    NodeBCsInput,
    ElemBCsInput,
    ReceiversOutput,
)
from gpu_solver import solver as gpu_solver

# Берём строитель сетки/материалов/приёмников как в test_rect
from test_rect import (
    _compute_gll_nodes_weights,
    _lagrange_derivative_matrix,
    _gll_1d_general,
    _gll_1d_deg1,
    build_rect_mesh_gll,
    build_materials,
    build_receivers_all_nodes,
)


@dataclass
class LambTestParams:
    # Геометрия (элементы)
    nx: int
    ny: int
    # Размер области
    Lx: float
    Ly: float
    # Полиномная степень (k=deg+1 узлов по 1D)
    deg: int
    # Время
    t_begin: float
    t_end: float
    steps: int  # 0 → авто по CFL
    frame_steps: int
    # Материал (плоская деформация)
    E: float
    nu: float
    rho: float
    alpha_x: float
    alpha_y: float
    # Пульс Рикера (вертикальная сила в центре верхней грани)
    ricker_freq: float
    ricker_amp: float
    # Параметр устойчивости CFL для автоматического шага по времени (если steps<=0)
    cfl: float = 0.35


def ricker_series(freq: float, amp: float, t_begin: float, t_end: float, steps: int) -> np.ndarray:
    """
    Временной ряд импульса Рикера длиной steps, на интервале [t_begin, t_end].
    Используем классическую формулу с центровкой в t0 = t_begin + 1/freq.
    """
    if steps <= 0:
        return np.zeros((0,), dtype=Float)
    t = np.linspace(t_begin, t_end, steps, dtype=Float)
    t0 = Float(t_begin + 1.0 / float(freq))
    pf2 = (np.pi * Float(freq)) ** 2
    x = (t - t0)
    return amp * (1.0 - 2.0 * pf2 * x * x) * np.exp(-pf2 * x * x)


def build_node_bcs_lamb(
    nodes_coords: np.ndarray,
    t_begin: float,
    t_end: float,
    steps: int,
    ricker_freq: float,
    ricker_amp: float,
) -> NodeBCsInput:
    """
    Узловые ГУ для задачи Лэмба:
      - nc0: Dirichlet u_x=0 на нижнем крае
      - nc1: Dirichlet u_y=0 на нижнем крае
      - nc2: Внешняя сила по Y = Ricker(t) в единственном узле — центре верхней грани
    """
    dim, N = nodes_coords.shape
    y = nodes_coords[1, :]
    x = nodes_coords[0, :]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    x_mid = float(0.5 * (float(np.min(x)) + float(np.max(x))))

    tol = 1e-12
    bottom_nodes = np.nonzero(np.abs(y - y_min) < tol)[0].astype(Index)
    top_nodes = np.nonzero(np.abs(y - y_max) < tol)[0].astype(Index)
    # Найдём узел на верхнем крае, ближайший к центру по X
    top_x = x[top_nodes]
    mid_idx_local = int(np.argmin(np.abs(top_x - x_mid)))
    source_node = int(top_nodes[mid_idx_local])

    # ГУ: 3 условия (ux=0 низ, uy=0 низ, fy(t) в source_node)
    nc = 3
    nodes_bc_types = np.array([2, 3, 1], dtype=Index)  # 2: clamp ux, 3: clamp uy, 1: force_y
    nodes_bc_begin = np.array([t_begin, t_begin, t_begin], dtype=Float)
    nodes_bc_end = np.array([t_end, t_end, t_end], dtype=Float)
    # Временные дискретизации условий: константы на низу, steps точек для Рикера
    nodes_bc_steps = np.array([1, 1, steps], dtype=Index)
    nodes_bc_nodes = np.array([bottom_nodes.size, bottom_nodes.size, 1], dtype=Index)
    nodes_bc_time_interp_type = np.array([0, 0, 1], dtype=Index)  # 0: ступенька; 1: линейная для Рикера

    # Отображения: узел -> позиция в списке узлов соответствующего ГУ
    pos_in_nc0 = {int(n): i for i, n in enumerate(bottom_nodes)}
    pos_in_nc1 = {int(n): i for i, n in enumerate(bottom_nodes)}
    pos_in_nc2 = {int(source_node): 0}

    # Подсчёт количества ГУ у каждого узла
    counts = np.zeros((N,), dtype=np.int64)
    for n in bottom_nodes:
        counts[int(n)] += 2  # ux, uy
    counts[source_node] += 1  # fy(t)

    nodes_bc_index_offsets = np.zeros((N + 1,), dtype=Index)
    np.cumsum(counts.astype(Index), out=nodes_bc_index_offsets[1:])
    nci = int(nodes_bc_index_offsets[-1])
    nodes_bc_index = np.zeros((2, nci), dtype=Index)

    # Заполнение index (по узлам в порядке 0..N-1)
    write_pos = 0
    for nid in range(N):
        if nid in pos_in_nc0:
            nodes_bc_index[0, write_pos] = 0
            nodes_bc_index[1, write_pos] = pos_in_nc0[nid]
            write_pos += 1
        if nid in pos_in_nc1:
            nodes_bc_index[0, write_pos] = 1
            nodes_bc_index[1, write_pos] = pos_in_nc1[nid]
            write_pos += 1
        if nid in pos_in_nc2:
            nodes_bc_index[0, write_pos] = 2
            nodes_bc_index[1, write_pos] = pos_in_nc2[nid]
            write_pos += 1

    # Данные ГУ: offsets по ГУ (время → узлы)
    nodes_bc_data_offsets = np.zeros((nc + 1,), dtype=Index)
    nodes_bc_data_offsets[0] = 0
    nodes_bc_data_offsets[1] = nodes_bc_data_offsets[0] + nodes_bc_nodes[0] * nodes_bc_steps[0]
    nodes_bc_data_offsets[2] = nodes_bc_data_offsets[1] + nodes_bc_nodes[1] * nodes_bc_steps[1]
    nodes_bc_data_offsets[3] = nodes_bc_data_offsets[2] + nodes_bc_nodes[2] * nodes_bc_steps[2]
    ncd = int(nodes_bc_data_offsets[-1])
    nodes_bc_data = np.zeros((ncd,), dtype=Float)

    # nc0/nc1 — нули по умолчанию (жёсткая фиксация в 0)
    # nc2 — временной ряд Рикера (в единственном узле)
    series = ricker_series(ricker_freq, ricker_amp, t_begin, t_end, steps)
    off2 = int(nodes_bc_data_offsets[2])
    nodes_bc_data[off2:off2 + steps] = series  # nodes=1, поэтому просто записываем подряд

    return NodeBCsInput(
        nc=Index(nc),
        nodes_bc_types=nodes_bc_types,
        nodes_bc_begin=nodes_bc_begin,
        nodes_bc_end=nodes_bc_end,
        nodes_bc_steps=nodes_bc_steps,
        nodes_bc_nodes=nodes_bc_nodes,
        nodes_bc_time_interp_type=nodes_bc_time_interp_type,
        nci=Index(nci),
        nodes_bc_index=nodes_bc_index,
        nodes_bc_index_offsets=nodes_bc_index_offsets,
        ncd=Index(ncd),
        nodes_bc_data=nodes_bc_data,
        nodes_bc_data_offsets=nodes_bc_data_offsets,
    )


def run_test_batch(param: LambTestParams):
    """
    Запускает расчет задачи Лэмба в batch-режиме (накапливает все кадры).
    Возвращает (U, V, F, nodes_coords, nx_nodes, ny_nodes).
    U: Float[steps, 2, N], V: Float[steps, 2, N], F: Float[steps, 2, N]
    """
    dim = 2
    k = param.deg + 1

    # 1) Prime types (1D)
    if param.deg == 1:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_deg1()
    else:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_general(param.deg)

    EPT = 1
    ept_nodes = np.array([k], dtype=Index)
    ept_dims = np.array([1], dtype=Index)
    ept_weights = gll_weights.astype(Float, copy=True)
    ept_weights_offsets = np.array([0, k], dtype=Index)
    ept_nabla_shapes = gll_nabla.astype(Float, copy=True)  # 1×k×k
    ept_nabla_shapes_offsets = np.array([0, k * k], dtype=Index)

    # 2) Геометрия и элементы
    nodes_coords, elements_nodes, elements_nodes_offsets, elements_types, elements_types_offsets = build_rect_mesh_gll(
        param.nx, param.ny, param.Lx, param.Ly, param.deg
    )
    nx_nodes = param.nx * param.deg + 1
    ny_nodes = param.ny * param.deg + 1
    N = nodes_coords.shape[1]
    E = param.nx * param.ny
    EN = elements_nodes.size

    # 3) Материалы (на уровне EN)
    material_props = build_materials(
        en=EN,
        E=param.E,
        nu=param.nu,
        rho=param.rho,
        alpha_x=param.alpha_x,
        alpha_y=param.alpha_y,
    )

    # 3.1) Автоматический расчёт шага/числа шагов по CFL при необходимости
    steps = int(param.steps)
    if steps <= 0:
        # минимальный шаг сетки по узлам (с учётом deg)
        hx = float(param.Lx) / float(param.nx * param.deg)
        hy = float(param.Ly) / float(param.ny * param.deg)
        h_min = min(hx, hy)
        lame_mu = float(param.E) / (2.0 * (1.0 + float(param.nu)))
        lame_la = float(param.E) * float(param.nu) / ((1.0 + float(param.nu)) * (1.0 - 2.0 * float(param.nu)))
        c_p = math.sqrt((lame_la + 2.0 * lame_mu) / float(param.rho))
        if c_p <= 0.0 or not math.isfinite(c_p):
            c_p = 1.0
        dt_cfl = float(param.cfl) * h_min / c_p
        total_time = float(param.t_end) - float(param.t_begin)
        if total_time <= 0.0:
            total_time = 1.0
        steps = max(1, int(math.ceil(total_time / dt_cfl)))

    # 4) Узловые ГУ (низ зажат, источник Рикера на верхнем центре)
    t_begin = float(param.t_begin)
    t_end = float(param.t_end)
    node_bcs = build_node_bcs_lamb(
        nodes_coords=nodes_coords,
        t_begin=t_begin,
        t_end=t_end,
        steps=steps,
        ricker_freq=param.ricker_freq,
        ricker_amp=param.ricker_amp,
    )

    # 5) Элементные ГУ — отсутствуют
    elem_bcs = ElemBCsInput(
        ec=Index(0),
        elems_condition_types=np.zeros((0,), dtype=Index),
        elems_condition_begin=np.zeros((0,), dtype=Float),
        elems_condition_end=np.zeros((0,), dtype=Float),
        elems_condition_steps=np.zeros((0,), dtype=Index),
        elems_condition_nodes=np.zeros((0,), dtype=Index),
        elems_condition_time_interp_type=np.zeros((0,), dtype=Index),
        eci=Index(0),
        elems_condition_index=np.zeros((2, 0), dtype=Index),
        elems_condition_index_offsets=np.zeros((EN + 1,), dtype=Index),
        ecd=Index(0),
        elems_condition_data=np.zeros((0,), dtype=Float),
        elems_condition_data_offsets=np.zeros((1,), dtype=Index),
    )

    # 6) Приемники
    receivers = build_receivers_all_nodes(
        t_begin=t_begin,
        t_end=t_end,
        steps=param.frame_steps,  # число кадров
        N=N,
    )

    # 7) Инициализирующее состояние
    ST = 4  # [u_x, u_y, v_x(1/2), v_y(1/2)]
    nodes_state = np.zeros((ST, N), dtype=Float)

    # 8) Интерфейс и запуск
    si = SolverInterface(
        time=SolverTime(
            time_begin=Float(t_begin),
            time_end=Float(t_end),
            time_steps=Index(steps)
        ),
        geom=NodesGeometryInput(
            dim=Index(dim),
            n=Index(N),
            dof=Index(dim * N),
            nodes_coords=nodes_coords
        ),
        state=NodesStateInput(
            st=Index(ST),
            nodes_state=nodes_state
        ),
        primes=PrimeElementTypesInput(
            ept=Index(EPT),
            ept_nodes=ept_nodes,
            ept_dims=ept_dims,
            epw=Index(ept_weights.size),
            ept_weights=ept_weights,
            ept_weights_offsets=ept_weights_offsets,
            epns=Index(ept_nabla_shapes.size),
            ept_nabla_shapes=ept_nabla_shapes,
            ept_nabla_shapes_offsets=ept_nabla_shapes_offsets,
        ),
        elems=CommonElementsInput(
            e=Index(E),
            et=Index(elements_types.size),
            elements_types=elements_types,
            elements_types_offsets=elements_types_offsets,
            en=Index(EN),
            elements_nodes=elements_nodes,
            elements_nodes_offsets=elements_nodes_offsets,
        ),
        materials=MaterialsInput(
            m=Index(material_props.shape[0]),
            material_props=material_props,
        ),
        node_bcs=node_bcs,
        elem_bcs=elem_bcs,
        receivers=receivers,
    )

    en = int(si.elems.en)
    print(f"[Test Lamb] EN={en}, N={N}, steps={steps}, frames={param.frame_steps}")


    # Сбор кадров через handler (batch-режим)
    frames_data: list[np.ndarray] = []
    frame_times: list[float] = []
    def handler(frame_time: float, frame_idx: int, recv_frame_flat: np.ndarray):
        frame_times.append(frame_time)
        frames_data.append(recv_frame_flat.copy())

    gpu_solver(si, frame_handler=handler)

    # Преобразуем приемники в [steps, components, N]
    if len(frames_data) == 0:
        frames = 0
        rec = np.zeros((0, int(si.receivers.receivers_components_offsets[1]), N), dtype=Float)
        T = np.zeros((0,), dtype=Float)
    else:
        frames = len(frames_data)
        rec = np.stack([fd.reshape(int(si.receivers.receivers_components_offsets[1]), N) for fd in frames_data], axis=0)
        T = np.array(frame_times, dtype=Float)
    U = rec[:, :2, :]
    V = rec[:, 2:4, :]
    F = rec[:, 4:, :]
    return U, V, F, nodes_coords, nx_nodes, ny_nodes


def run_test_realtime(param: LambTestParams):
    """
    Запускает расчет задачи Лэмба в real-time режиме с анимацией через потоки.
    Возвращает интерфейс задачи для последующего анализа.
    """
    dim = 2
    k = param.deg + 1

    # 1) Prime types (1D)
    if param.deg == 1:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_deg1()
    else:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_general(param.deg)

    EPT = 1
    ept_nodes = np.array([k], dtype=Index)
    ept_dims = np.array([1], dtype=Index)
    ept_weights = gll_weights.astype(Float, copy=True)
    ept_weights_offsets = np.array([0, k], dtype=Index)
    ept_nabla_shapes = gll_nabla.astype(Float, copy=True)
    ept_nabla_shapes_offsets = np.array([0, k * k], dtype=Index)

    # 2) Геометрия и элементы
    nodes_coords, elements_nodes, elements_nodes_offsets, elements_types, elements_types_offsets = build_rect_mesh_gll(
        param.nx, param.ny, param.Lx, param.Ly, param.deg
    )
    nx_nodes = param.nx * param.deg + 1
    ny_nodes = param.ny * param.deg + 1
    N = nodes_coords.shape[1]
    E = param.nx * param.ny
    EN = elements_nodes.size

    # 3) Материалы
    material_props = build_materials(
        en=EN,
        E=param.E,
        nu=param.nu,
        rho=param.rho,
        alpha_x=param.alpha_x,
        alpha_y=param.alpha_y,
    )

    # 3.1) Автоматический расчёт шага
    steps = int(param.steps)
    if steps <= 0:
        hx = float(param.Lx) / float(param.nx * param.deg)
        hy = float(param.Ly) / float(param.ny * param.deg)
        h_min = min(hx, hy)
        lame_mu = float(param.E) / (2.0 * (1.0 + float(param.nu)))
        lame_la = float(param.E) * float(param.nu) / ((1.0 + float(param.nu)) * (1.0 - 2.0 * float(param.nu)))
        c_p = math.sqrt((lame_la + 2.0 * lame_mu) / float(param.rho))
        if c_p <= 0.0 or not math.isfinite(c_p):
            c_p = 1.0
        dt_cfl = float(param.cfl) * h_min / c_p
        total_time = float(param.t_end) - float(param.t_begin)
        if total_time <= 0.0:
            total_time = 1.0
        steps = max(1, int(math.ceil(total_time / dt_cfl)))

    # 4) Узловые ГУ
    t_begin = float(param.t_begin)
    t_end = float(param.t_end)
    node_bcs = build_node_bcs_lamb(
        nodes_coords=nodes_coords,
        t_begin=t_begin,
        t_end=t_end,
        steps=steps,
        ricker_freq=param.ricker_freq,
        ricker_amp=param.ricker_amp,
    )

    # 5) Элементные ГУ — отсутствуют
    elem_bcs = ElemBCsInput(
        ec=Index(0),
        elems_condition_types=np.zeros((0,), dtype=Index),
        elems_condition_begin=np.zeros((0,), dtype=Float),
        elems_condition_end=np.zeros((0,), dtype=Float),
        elems_condition_steps=np.zeros((0,), dtype=Index),
        elems_condition_nodes=np.zeros((0,), dtype=Index),
        elems_condition_time_interp_type=np.zeros((0,), dtype=Index),
        eci=Index(0),
        elems_condition_index=np.zeros((2, 0), dtype=Index),
        elems_condition_index_offsets=np.zeros((EN + 1,), dtype=Index),
        ecd=Index(0),
        elems_condition_data=np.zeros((0,), dtype=Float),
        elems_condition_data_offsets=np.zeros((1,), dtype=Index),
    )

    # 6) Приемники
    receivers = build_receivers_all_nodes(
        t_begin=t_begin,
        t_end=t_end,
        steps=param.frame_steps,
        N=N,
    )

    # 7) Инициализирующее состояние
    ST = 4
    nodes_state = np.zeros((ST, N), dtype=Float)

    # 8) Интерфейс
    si = SolverInterface(
        time=SolverTime(
            time_begin=Float(t_begin),
            time_end=Float(t_end),
            time_steps=Index(steps)
        ),
        geom=NodesGeometryInput(
            dim=Index(dim),
            n=Index(N),
            dof=Index(dim * N),
            nodes_coords=nodes_coords
        ),
        state=NodesStateInput(
            st=Index(ST),
            nodes_state=nodes_state
        ),
        primes=PrimeElementTypesInput(
            ept=Index(EPT),
            ept_nodes=ept_nodes,
            ept_dims=ept_dims,
            epw=Index(ept_weights.size),
            ept_weights=ept_weights,
            ept_weights_offsets=ept_weights_offsets,
            epns=Index(ept_nabla_shapes.size),
            ept_nabla_shapes=ept_nabla_shapes,
            ept_nabla_shapes_offsets=ept_nabla_shapes_offsets,
        ),
        elems=CommonElementsInput(
            e=Index(E),
            et=Index(elements_types.size),
            elements_types=elements_types,
            elements_types_offsets=elements_types_offsets,
            en=Index(EN),
            elements_nodes=elements_nodes,
            elements_nodes_offsets=elements_nodes_offsets,
        ),
        materials=MaterialsInput(
            m=Index(material_props.shape[0]),
            material_props=material_props,
        ),
        node_bcs=node_bcs,
        elem_bcs=elem_bcs,
        receivers=receivers,
    )

    print(f"[Test Lamb RT] EN={int(si.elems.en)}, N={N}, steps={steps}, frames={param.frame_steps}")

    # === Real-time режим с потоками ===
    
    stop_event = threading.Event()
    frames: Queue[Tuple[float, int, np.ndarray]] = Queue(maxsize=1)  # backpressure
    
    # Геометрия для визуализации
    X = nodes_coords[0, :].reshape(ny_nodes, nx_nodes)
    Y = nodes_coords[1, :].reshape(ny_nodes, nx_nodes)
    
    # Текущий кадр для отображения
    n_components = int(si.receivers.receivers_components_offsets[1] - si.receivers.receivers_components_offsets[0])
    frame_current = np.zeros((n_components, N), dtype=Float)
    
    # Создаём фигуру
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Задача Лэмба: |u| (real-time)')
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))
    
    # Начальное изображение
    Umag_init = np.zeros((ny_nodes, nx_nodes), dtype=Float)
    qm = ax.pcolormesh(X, Y, Umag_init, shading='gouraud', cmap='viridis', vmin=0.0, vmax=1.0)
    cb = fig.colorbar(qm, ax=ax)
    cb.set_label('|u| (m)')
    
    def handler(frame_time: float, frame_idx: int, recv_frame_flat: np.ndarray):
        """Handler помещает кадр в очередь (с блокировкой для backpressure)"""
        frames.put((frame_time, frame_idx, recv_frame_flat.copy()), block=True)
    
    # Рабочий поток для солвера
    worker = threading.Thread(
        target=gpu_solver, 
        args=(si,), 
        kwargs={'frame_handler': handler, 'stop_event': stop_event},
        daemon=True
    )
    worker.start()
    
    def update(_):
        """Функция обновления анимации (вызывается matplotlib)"""
        try:
            # Пытаемся получить новый кадр без блокировки
            frame_time, frame_idx, frame_data = frames.get_nowait()
            
            # Преобразуем в [components, N]
            frame_current[:] = frame_data.reshape(n_components, N)
            
            # Вычисляем |u|
            Ux = frame_current[0, :].reshape(ny_nodes, nx_nodes)
            Uy = frame_current[1, :].reshape(ny_nodes, nx_nodes)
            Umag = np.sqrt(Ux * Ux + Uy * Uy)
            
            # Обновляем изображение
            vmin = float(np.nanmin(Umag))
            vmax = float(np.nanmax(Umag))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1e-12
            
            qm.set_array(Umag.ravel())
            qm.set_clim(vmin=vmin, vmax=vmax)
            cb.update_normal(qm)
            cb.set_label(f'|u| [{vmin:.3g}; {vmax:.3g}] t={frame_time:.4f}s')
            
        except Empty:
            pass  # Нет нового кадра — показываем предыдущий
        
        return [qm]
    
    def on_close(_):
        """При закрытии окна останавливаем вычисления"""
        stop_event.set()
    
    fig.canvas.mpl_connect("close_event", on_close)
    
    # Запускаем анимацию
    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.show()
    
    # Останавливаем worker thread
    stop_event.set()
    worker.join(timeout=2.0)
    
    return si


def animate_field(U: np.ndarray, nodes_coords: np.ndarray, nx_nodes: int, ny_nodes: int, interval_ms: int = 30):
    """
    Анимация величины смещения |u| на регулярной сетке (k=2 → прямоугольная решетка).
    """
    steps, _, N = U.shape
    assert nodes_coords.shape[1] == N
    X = nodes_coords[0, :].reshape(ny_nodes, nx_nodes)
    Y = nodes_coords[1, :].reshape(ny_nodes, nx_nodes)

    Ux = U[:, 0, :].reshape(steps, ny_nodes, nx_nodes)
    Uy = U[:, 1, :].reshape(steps, ny_nodes, nx_nodes)
    Umag = np.sqrt(Ux * Ux + Uy * Uy)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Задача Лэмба: |u|')
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))

    qm = ax.pcolormesh(X, Y, Umag[0], shading='gouraud', cmap='viridis')
    cb = fig.colorbar(qm, ax=ax)
    cb.set_label('|u|')

    def update(frame: int):
        data = Umag[frame]
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 0.0, 0.0
        if vmax <= vmin:
            vmax = vmin + 1e-12
        qm.set_array(data.ravel())
        qm.set_clim(vmin=vmin, vmax=vmax)
        cb.update_normal(qm)
        cb.set_label(f'|u| [{vmin:.3g}; {vmax:.3g}] ({frame})')
        return (qm,)

    anim = FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
    plt.show()

deg = 2


def main_batch():
    """Batch-режим: накапливаем все кадры, затем показываем анимацию"""
    p = LambTestParams(
        nx=240,
        ny=120,
        Lx=5.0,
        Ly=2.0,
        deg=deg,
        t_begin=0.0,
        t_end=1.0,
        steps=1000,  # число временных шагов расчёта
        frame_steps=26,  # число кадров для анимации
        E=1.0e5,
        nu=0.25,
        rho=1000.0,
        alpha_x=1.0,
        alpha_y=1.0,
        ricker_freq=30.0,
        ricker_amp=100.0,
        cfl=0.35,
    )
    U, V, F, nodes_coords, nx_nodes, ny_nodes = run_test_batch(p)
    animate_field(F, nodes_coords, nx_nodes, ny_nodes, interval_ms=100)

multyplicator = 10

def main_realtime():
    """Real-time режим: показываем анимацию по мере вычисления"""
    p = LambTestParams(
        nx=120,
        ny=60,
        Lx=5.0,
        Ly=2.0,
        deg=deg,
        t_begin=0.0,
        t_end=1.0*multyplicator,
        steps=2000*multyplicator,  # число временных шагов расчёта
        frame_steps=1,  # число кадров для анимации
        E=1.0e5,
        nu=0.25,
        rho=1000.0,
        alpha_x=1.0,
        alpha_y=1.0,
        ricker_freq=30.0,
        ricker_amp=100.0,
        cfl=0.35,
    )
    run_test_realtime(p)


if __name__ == "__main__":
    import sys
    
    # Выбираем режим через аргумент командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        print("[Main] Запуск в batch-режиме...")
        main_batch()
    else:
        print("[Main] Запуск в real-time режиме...")
        main_realtime()


