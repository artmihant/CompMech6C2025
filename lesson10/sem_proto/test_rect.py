from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import os
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from gpu_solver import solver as gpu_solver
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


@dataclass
class RectTestParams:
    # Геометрия (элементы)
    nx: int  # число элементов по X
    ny: int   # число элементов по Y
    # Размер области
    Lx: float
    Ly: float
    # Полиномная степень (k=deg+1 узлов по 1D). В стенде фиксируем линейный элемент (deg=1, k=2).
    deg: int
    # Время
    t_begin: float
    t_end: float
    steps: int
    # Материал (плоская деформация)
    E: float
    nu: float
    rho: float
    alpha_x: float
    alpha_y: float
    # Нагрузка (равномерная по верхнему краю, вдоль Y)
    top_fy: float
    # Параметр устойчивости CFL для автоматического шага по времени (если steps<=0)
    cfl: float = 0.35
def _compute_gll_nodes_weights(deg: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Узлы и веса GLL для степени 'deg' (число узлов k=deg+1).
    Веса: w_i = 2 / (deg*(deg+1) * P_deg(x_i)^2), где P_deg — полином Лежандра степени deg.
    """
    from numpy.polynomial.legendre import legroots, legder, legval
    if deg < 1:
        raise ValueError("deg должен быть >= 1")
    Pn = [0.0] * deg + [1.0]
    dPn = legder(Pn)
    interior = legroots(dPn)
    nodes = np.empty((deg + 1,), dtype=Float)
    nodes[0] = -1.0
    if interior.size > 0:
        nodes[1:-1] = np.array(sorted(interior.tolist()), dtype=Float)
    nodes[-1] = 1.0
    Pn_vals = legval(nodes, Pn).astype(Float)
    weights = (2.0 / (deg * (deg + 1.0) * (Pn_vals * Pn_vals))).astype(Float)
    return nodes, weights


def _lagrange_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Матрица D, D[i,j] = dL_j/dξ(ξ_i) для узлов 'nodes' (барицентрическая формула).
    """
    k = int(nodes.shape[0])
    lam = np.ones((k,), dtype=Float)
    for j in range(k):
        denom = 1.0
        xj = nodes[j]
        for m in range(k):
            if m == j:
                continue
            denom *= (xj - nodes[m])
        lam[j] = 1.0 / denom
    D = np.zeros((k, k), dtype=Float)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            D[i, j] = lam[j] / (lam[i] * (nodes[i] - nodes[j]))
        D[i, i] = -np.sum(D[i, :])
    return D


def _gll_1d_general(deg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Общий 1D GLL: (nodes[k], weights[k], nabla[1*k*k]) для произвольной степени.
    nabla упакован как [mu, nu] (d=0), где значение = dL_mu/dξ @ ξ_nu.
    """
    nodes, weights = _compute_gll_nodes_weights(deg)
    D = _lagrange_derivative_matrix(nodes)  # D[nu, mu]
    nabla = D.T.astype(Float, copy=False).reshape(-1)
    return nodes.astype(Float, copy=False), weights.astype(Float, copy=False), nabla




def _gll_1d_deg1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Узлы/веса/производные базисов Лагранжа на узлах для 1D GLL при deg=1 (k=2).
    Возвращает:
      - nodes: Float[2] = [-1, +1]
      - weights: Float[2] = [1, 1]
      - nabla_shapes: Float[1*2*2] = [dL0/dξ@ξ0, dL0/dξ@ξ1, dL1/dξ@ξ0, dL1/dξ@ξ1]
        порядок хранения: d=0 блок, затем mu (базис), затем nu (узел)
    """
    nodes = np.array([-1.0, 1.0], dtype=Float)
    weights = np.array([1.0, 1.0], dtype=Float)
    # L0 = (1-ξ)/2, L1=(1+ξ)/2 → L0'=-1/2, L1'=+1/2 (константы).
    # Память как [mu, nu] (см. cpu_device_utils.local_grad индексацию).
    nabla = np.array([
        -0.5, -0.5,  # mu=0 (L0'), nu=0..1
         0.5,  0.5,  # mu=1 (L1'), nu=0..1
    ], dtype=Float)
    return nodes, weights, nabla


def build_rect_mesh_gll(nx: int, ny: int, Lx: float, Ly: float, deg: int):
    """
    Прямоугольная регулярная сетка GLL для степени 'deg' (k=deg+1 узлов 1D на элемент).
    Глобальные узлы формируются без дублирования на стыках:
      Nx_nodes = nx*deg + 1, Ny_nodes = ny*deg + 1.
    Локальная нумерация в элементе: x-индекс меняется быстрее.
    Возвращает:
      - nodes_coords: Float[2,N]
      - elements_nodes: Index[en], en = (k*k)*E
      - elements_nodes_offsets: Index[E+1]
      - elements_types: Index[2*E]  (по два 1D-типа на элемент: (x,y))
      - elements_types_offsets: Index[E+1]
    """
    if deg < 1:
        raise ValueError("deg должен быть >= 1")
    k = deg + 1
    dim = 2
    nx_nodes = nx * deg + 1
    ny_nodes = ny * deg + 1
    N = nx_nodes * ny_nodes

    # 1D GLL узлы на [-1, 1]
    gll_xi, _ = _compute_gll_nodes_weights(deg)
    dx = Float(Lx) / float(nx)
    dy = Float(Ly) / float(ny)

    xs = np.empty((nx_nodes,), dtype=Float)
    ys = np.empty((ny_nodes,), dtype=Float)

    # X-координаты (склейка без дубликатов)
    write = 0
    for i in range(nx):
        xl = Float(i) * dx
        xr = Float(i + 1) * dx
        mapped = xl + (gll_xi + 1.0) * (xr - xl) * 0.5
        if i == 0:
            xs[write:write + k] = mapped
            write += k
        else:
            xs[write:write + (k - 1)] = mapped[1:]
            write += (k - 1)

    # Y-координаты (склейка без дубликатов)
    write = 0
    for j in range(ny):
        yb = Float(j) * dy
        yt = Float(j + 1) * dy
        mapped = yb + (gll_xi + 1.0) * (yt - yb) * 0.5
        if j == 0:
            ys[write:write + k] = mapped
            write += k
        else:
            ys[write:write + (k - 1)] = mapped[1:]
            write += (k - 1)

    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes_coords = np.zeros((dim, N), dtype=Float)
    nodes_coords[0, :] = X.reshape(-1)
    nodes_coords[1, :] = Y.reshape(-1)

    def gid(ix: int, iy: int) -> int:
        return iy * nx_nodes + ix

    E = nx * ny
    en_per_elem = k * k
    elements_nodes = np.zeros((E * en_per_elem,), dtype=Index)
    elements_nodes_offsets = np.zeros((E + 1,), dtype=Index)
    elements_types = np.zeros((E * dim,), dtype=Index)
    elements_types_offsets = np.zeros((E + 1,), dtype=Index)

    en_off = 0
    et_off = 0
    eidx = 0
    for j in range(ny):
        for i in range(nx):
            # Диапазоны локальных индексов в глобальной сетке
            ix0 = i * deg
            iy0 = j * deg
            w = 0
            for vy in range(k):
                gy = iy0 + vy
                for ux in range(k):
                    gx = ix0 + ux
                    elements_nodes[en_off + w] = gid(gx, gy)
                    w += 1
            elements_nodes_offsets[eidx] = en_off
            en_off += en_per_elem

            # Составной тип: (1D_x, 1D_y) → оба указывают на тип #0
            elements_types[et_off + 0] = 0
            elements_types[et_off + 1] = 0
            elements_types_offsets[eidx] = et_off
            et_off += dim

            eidx += 1

    elements_nodes_offsets[E] = en_off
    elements_types_offsets[E] = et_off

    return nodes_coords, elements_nodes, elements_nodes_offsets, elements_types, elements_types_offsets


def build_materials(en: int, E: float, nu: float, rho: float, alpha_x: float, alpha_y: float) -> np.ndarray:
    """
    Возвращает Float[M,EN] с порядком:
      [0]=rho, [1]=lambda, [2]=mu, [3]=alpha_x, [4]=alpha_y
    """
    lame_mu = E / (2.0 * (1.0 + nu))
    lame_la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    M = 5
    out = np.zeros((M, en), dtype=Float)
    out[0, :] = rho
    out[1, :] = lame_la
    out[2, :] = lame_mu
    out[3, :] = alpha_x
    out[4, :] = alpha_y
    return out


def build_node_bcs(
    nodes_coords: np.ndarray,
    t_begin: float,
    t_end: float,
    steps: int,
    top_fy: float,
):
    """
    Узловые ГУ:
      - nc0: Dirichlet u_x=0 на нижнем крае
      - nc1: Dirichlet u_y=0 на нижнем крае
      - nc2: Внешняя сила по X = top_fy на верхнем крае
    """
    dim, N = nodes_coords.shape
    y = nodes_coords[1, :]
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    tol = 1e-12
    bottom_nodes = np.nonzero(np.abs(y - y_min) < tol)[0].astype(Index)
    top_nodes = np.nonzero(np.abs(y - y_max) < tol)[0].astype(Index)

    nc = 3
    nodes_bc_types = np.array([2, 3, 1], dtype=Index)  # u_x clamp, u_y clamp, force_y
    nodes_bc_begin = np.array([t_begin, t_begin, t_begin], dtype=Float)
    nodes_bc_end = np.array([t_end, t_end, t_end], dtype=Float)
    nodes_bc_steps = np.array([1, 1, 1], dtype=Index)  # константы по времени
    nodes_bc_nodes = np.array([bottom_nodes.size, bottom_nodes.size, top_nodes.size], dtype=Index)
    nodes_bc_time_interp_type = np.array([0, 0, 0], dtype=Index)  # ступенька

    # Индексация по узлам: offsets[N+1], index[2*NCI] = [nc_id..., space_pos...]
    # Сначала построим отображения узел → позиция в BC
    pos_in_nc0 = {int(n): i for i, n in enumerate(bottom_nodes)}
    pos_in_nc1 = {int(n): i for i, n in enumerate(bottom_nodes)}
    pos_in_nc2 = {int(n): i for i, n in enumerate(top_nodes)}

    # Посчитаем, сколько BC у каждого узла
    counts = np.zeros((N,), dtype=np.int64)
    for n in bottom_nodes:
        counts[int(n)] += 2  # ux, uy
    for n in top_nodes:
        counts[int(n)] += 1  # fx

    nodes_bc_index_offsets = np.zeros((N + 1,), dtype=Index)
    np.cumsum(counts.astype(Index), out=nodes_bc_index_offsets[1:])
    nci = int(nodes_bc_index_offsets[-1])
    nodes_bc_index = np.zeros((2, nci), dtype=Index)

    # Заполнение index
    write_pos = 0
    for nid in range(N):
        # ux clamp
        if nid in pos_in_nc0:
            nodes_bc_index[0, write_pos] = 0
            nodes_bc_index[1, write_pos] = pos_in_nc0[nid]
            write_pos += 1
        # uy clamp
        if nid in pos_in_nc1:
            nodes_bc_index[0, write_pos] = 1
            nodes_bc_index[1, write_pos] = pos_in_nc1[nid]
            write_pos += 1
        # fx top
        if nid in pos_in_nc2:
            nodes_bc_index[0, write_pos] = 2
            nodes_bc_index[1, write_pos] = pos_in_nc2[nid]
            write_pos += 1

    # Данные ГУ: порядок «время → узлы», по 1 компоненте
    nodes_bc_data_offsets = np.zeros((nc + 1,), dtype=Index)
    nodes_bc_data_offsets[0] = 0
    nodes_bc_data_offsets[1] = nodes_bc_data_offsets[0] + nodes_bc_nodes[0] * nodes_bc_steps[0]
    nodes_bc_data_offsets[2] = nodes_bc_data_offsets[1] + nodes_bc_nodes[1] * nodes_bc_steps[1]
    nodes_bc_data_offsets[3] = nodes_bc_data_offsets[2] + nodes_bc_nodes[2] * nodes_bc_steps[2]
    ncd = int(nodes_bc_data_offsets[-1])
    nodes_bc_data = np.zeros((ncd,), dtype=Float)
    # nc0/nc1 — нули (Дирихле в 0)
    # nc2 — константа top_fy
    off2 = int(nodes_bc_data_offsets[2])
    nodes_bc_data[off2:off2 + int(nodes_bc_nodes[2])] = top_fy

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


def build_receivers_all_nodes(t_begin: float, t_end: float, steps: int, N: int) -> ReceiversOutput:
    """
    Один приемник, пишет [u_x,u_y,v_x,v_y,a_x,a_y] на всех узлах.
    steps трактуем как число кадров (frames), а буфер данных — один кадр.
    """
    r = 1
    receivers_types = np.zeros((r,), dtype=Index)
    receivers_begin = np.array([t_begin], dtype=Float)
    receivers_end = np.array([t_end], dtype=Float)
    receivers_steps = np.array([steps], dtype=Index)

    components = 6  # [u_x,u_y,v_x,v_y,a_x,a_y]

    receivers_components_offsets = np.array([0, components], dtype=Index)
    receivers_components = np.array([0, 1, 2, 3, 4, 5], dtype=Index)
    rc = int(receivers_components.size)

    # Индексация по узлам: каждый узел участвует в приемнике r0 с позицией = nid
    receivers_index_offsets = np.arange(0, N + 1, dtype=Index)
    ri = N
    receivers_index = np.zeros((2,ri), dtype=Index)
    for nid in range(N):
        receivers_index[0, nid] = 0  # recv id
        receivers_index[1, nid] = nid  # space_pos

    # Данные: один кадр = components * N
    rd_per_receiver = components * int(N)
    receivers_data_offsets = np.array([0, rd_per_receiver], dtype=Index)
    rd = rd_per_receiver
    receivers_data = np.zeros((rd,), dtype=Float)

    return ReceiversOutput(
        r=Index(r),
        receivers_types=receivers_types,
        receivers_begin=receivers_begin,
        receivers_end=receivers_end,
        receivers_steps=receivers_steps,
        rc=Index(rc),
        receivers_components=receivers_components,
        receivers_components_offsets=receivers_components_offsets,
        ri=Index(ri),
        receivers_index=receivers_index,
        receivers_index_offsets=receivers_index_offsets,
        rd=Index(rd),
        receivers_data=receivers_data,
        receivers_data_offsets=receivers_data_offsets,
    )


def run_test(p: RectTestParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Запускает расчет и возвращает (UxUy_over_time, nodes_coords, nx_nodes, ny_nodes).
    UxUy_over_time: Float[steps, 2, N]
    """
    dim = 2
    k = p.deg + 1

    # 1) Prime types (1D)
    if p.deg == 1:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_deg1()
    else:
        gll_nodes, gll_weights, gll_nabla = _gll_1d_general(p.deg)
    EPT = 1
    ept_nodes = np.array([k], dtype=Index)
    ept_dims = np.array([1], dtype=Index)
    ept_weights = gll_weights.astype(Float, copy=True)
    ept_weights_offsets = np.array([0, k], dtype=Index)
    ept_nabla_shapes = gll_nabla.astype(Float, copy=True)  # 1×k×k
    ept_nabla_shapes_offsets = np.array([0, k * k], dtype=Index)

    # 2) Геометрия и элементы
    nodes_coords, elements_nodes, elements_nodes_offsets, elements_types, elements_types_offsets = build_rect_mesh_gll(
        p.nx, p.ny, p.Lx, p.Ly, p.deg
    )
    nx_nodes = p.nx * p.deg + 1
    ny_nodes = p.ny * p.deg + 1
    N = nodes_coords.shape[1]
    E = p.nx * p.ny
    EN = elements_nodes.size

    # 3) Материалы (на уровне EN)
    material_props = build_materials(
        en=EN,
        E=p.E,
        nu=p.nu,
        rho=p.rho,
        alpha_x=p.alpha_x,
        alpha_y=p.alpha_y,
    )

    # 3.1) Автоматический расчёт шага/числа шагов по CFL при необходимости
    steps = int(p.steps)
    if steps <= 0:
        # минимальный шаг сетки по узлам (с учётом deg)
        hx = float(p.Lx) / float(p.nx * p.deg)
        hy = float(p.Ly) / float(p.ny * p.deg)
        h_min = min(hx, hy)
        lame_mu = float(p.E) / (2.0 * (1.0 + float(p.nu)))
        lame_la = float(p.E) * float(p.nu) / ((1.0 + float(p.nu)) * (1.0 - 2.0 * float(p.nu)))
        c_p = math.sqrt((lame_la + 2.0 * lame_mu) / float(p.rho))
        if c_p <= 0.0 or not math.isfinite(c_p):
            c_p = 1.0
        dt_cfl = float(p.cfl) * h_min / c_p
        total_time = float(p.t_end) - float(p.t_begin)
        if total_time <= 0.0:
            total_time = 1.0
        steps = max(1, int(math.ceil(total_time / dt_cfl)))
        print(steps)
    # локальная копия времени шага для построения БС/приёмников
    t_begin = float(p.t_begin)
    t_end = float(p.t_end)
    # 4) Узловые ГУ
    node_bcs = build_node_bcs(
        nodes_coords=nodes_coords,
        t_begin=t_begin,
        t_end=t_end,
        steps=steps,
        top_fy=p.top_fy,
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
        steps=steps,
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

    # Сбор кадров через handler
    frames_data: list[np.ndarray] = []
    frame_times: list[float] = []
    def handler(frame_time: float, frame_idx: int, recv_frame_flat: np.ndarray):
        # recv_frame_flat: SoA [components → nodes]
        frame_times.append(frame_time)
        frames_data.append(recv_frame_flat.copy())

    gpu_solver(si, frame_handler=handler)

    # Преобразуем собранные кадры в [frames, components, N]
    if len(frames_data) == 0:
        frames = 0
        rec = np.zeros((0, 6, N), dtype=Float)
    else:
        frames = len(frames_data)
        rec = np.stack([fd.reshape(6, N) for fd in frames_data], axis=0)
    U = rec[:, :2, :]
    V = rec[:, 2:4, :]
    F = rec[:, 4:, :]
    T = np.array(frame_times, dtype=Float)

    return U, V, F, T, nodes_coords, nx_nodes, ny_nodes


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
    ax.set_title('Поле |u| (величина смещения)')
    ax.set_xlim(float(np.min(X)), float(np.max(X)))
    ax.set_ylim(float(np.min(Y)), float(np.max(Y)))

    # Используем pcolormesh по сетке узлов
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


def main():
    p = RectTestParams(
        nx=99,
        ny=99,
        Lx=2.0,
        Ly=2.0,
        deg=1,
        t_begin=0.0,
        t_end=1.0,
        steps=50,  # число кадров (пример)
        E=1.0e5,
        nu=0.25,
        rho=1000.0,
        alpha_x=1.0,
        alpha_y=1.0,
        top_fy=1.0,
        cfl=0.35,
    )
    U, V, F, T, nodes_coords, nx_nodes, ny_nodes = run_test(p)
    animate_field(U, nodes_coords, nx_nodes, ny_nodes, interval_ms=100)


if __name__ == "__main__":
    main()


