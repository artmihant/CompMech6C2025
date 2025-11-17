from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np

from numba import cuda

from interface import (
    Index,
    Float,
    SolverInterface,
    validate_interface,
)

from gpu_device_utils import (
    empty_array,
    empty_array_kernel,
    matmul_inplace,
    get_time_interpol_bspline,
    local_div,
    local_grad,
    report_cuda_mem,
)

# ===== GPU kernels =====

DIM = 2
THREADS_COUNT = 32 # Число потоков на блок: должно быть меньше максимального числа узлов в элементе и делиться на 32.

@cuda.jit
def init_element_kernel(
    e_count: Index, # int: число элементов
    en_count: Index, # int: число узлов во всех элементах соввокупно

    d_elem_nodes: np.ndarray, # int[EN]: узлы элементов
    d_elem_nodes_offsets: np.ndarray, # int[E+1]: оффсеты узлов элементов

    d_nodes_coords: np.ndarray, # Float[DIM, N]: координаты узлов (SoA: сначала все X, затем Y, ... в виде 2D)

    d_ept_nodes: np.ndarray, # int[EPT]: число узлов в простых типах элементов 
    d_ept_dims: np.ndarray, # int[EPT]: размерности простых типов элементов 
    d_ept_nabla: np.ndarray, # float[EPNS, EPT]: градиенты форм простых типов элементов
    d_ept_nabla_offsets: np.ndarray, # int[EPT+1]: оффсеты градиентов форм простых типов элементов
    d_elem_types: np.ndarray, # int[ET]: составные типы элементов
    d_elem_types_offsets: np.ndarray, # int[E+1]: оффсеты типов элементов

    d_ept_weights: np.ndarray, # Float[EPW]: Веса квадратур простых элементов
    d_ept_weights_offsets: np.ndarray, # Index[EPT+1]: Оффсеты весов

    d_en_volumes: np.ndarray, # float[EN]: массив записи результата: обьемы (глобальный аналог весов)
    d_en_yacobies: np.ndarray, # float[dim*dim*EN]: массив записи результата: матрица обратных якобианов
):

    """ Ядро инициализации записывает en инвертированных якобианов и en обьемов (глобальных весов)"""

    """Вычисляем позицию"""
    eid = cuda.blockIdx.x

    if eid >= e_count:
        return

    elem_nodes_offset = d_elem_nodes_offsets[eid]

    nodes_in_element = d_elem_nodes_offsets[eid+1]-elem_nodes_offset

    nu: int = cuda.threadIdx.x

    if nu >= nodes_in_element:
        return 

    enid = elem_nodes_offset+nu
    if enid >= en_count:
        return

    nid = d_elem_nodes[enid] # индекс глобального узла элемента

    """Извлечь и расшарить инфу об кооординатах"""

    s_nodes_coords = cuda.shared.array((DIM, THREADS_COUNT), Float)

    for d in range(DIM):
        s_nodes_coords[d, nu] = d_nodes_coords[d, nid]

    cuda.syncthreads() 

    inv_yacobi = cuda.local.array((DIM, DIM), dtype=np.float64)
    empty_array(inv_yacobi)

    local_grad(
        eid, nu, 
        s_nodes_coords, DIM,         
        d_ept_nodes, 
        d_ept_dims,
        d_ept_nabla,
        d_ept_nabla_offsets,
        d_elem_types,
        d_elem_types_offsets,
        inv_yacobi
    )

    ## Переворачиваем in_yacobi и записываем для dim = 2 случая. 

    yacobian = inv_yacobi[0,0] * inv_yacobi[1,1] - inv_yacobi[1,0] * inv_yacobi[0,1]

    d_en_yacobies[0,0,enid] =  inv_yacobi[1,1]/yacobian
    d_en_yacobies[0,1,enid] = -inv_yacobi[0,1]/yacobian
    d_en_yacobies[1,0,enid] = -inv_yacobi[1,0]/yacobian
    d_en_yacobies[1,1,enid] =  inv_yacobi[0,0]/yacobian

    elem_types_offset = d_elem_types_offsets[eid]
    t_types_in_element = d_elem_types_offsets[eid+1] - elem_types_offset

    en_volume = yacobian
    mod_mul = 1

    for k_type_id in range(t_types_in_element):
        type_id = d_elem_types[elem_types_offset+k_type_id]

        ept_weights_offset = d_ept_weights_offsets[type_id]
        nodes_k = d_ept_nodes[type_id]

        nu_k = (nu // mod_mul) % nodes_k

        ept_weight_k = d_ept_weights[ept_weights_offset+nu_k]

        en_volume *= ept_weight_k
        mod_mul *= nodes_k

    d_en_volumes[enid] = en_volume


@cuda.jit
def inner_forces_element_kernel(

    d_ept_nodes: np.ndarray, # int[EPT]: число узлов в простых типах элементов 
    d_ept_dims: np.ndarray, # int[EPT]: размерности простых типов элементов 
    
    d_ept_nabla: np.ndarray, # float[EPNS>EPT]: градиенты форм простых типов элементов
    d_ept_nabla_offsets: np.ndarray, # int[EPT+1]: оффсеты градиентов форм простых типов элементов
    
    e_count: Index, # int: число элементов

    d_elem_types: np.ndarray, # int[ET>E]: составные типы элементов
    d_elem_types_offsets: np.ndarray, # int[E+1]: оффсеты типов элементов

    d_elem_nodes: np.ndarray, # int[EN>E]: узлы элементов
    d_elem_nodes_offsets: np.ndarray, # int[E+1]: оффсеты узлов элементов

    d_en_volumes: np.ndarray, # float[EN]: интегралы функций форм узлах элементов.
    d_en_yacobies: np.ndarray, # float[dim,dim,EN]: предрассчитанная обратная матрица Якоби в узлах элементов

    d_nodes_state: np.ndarray, # float[ST,N]: перемещения и скорости узлов (основное кинматическое состояние задачи), SoA (ux..., uy..., , vx..., vy...)
    d_nodes_sub_state: np.ndarray, # float[dim, N]: Внутренние упругие силы

    d_mat_props: np.ndarray, # float[MP,EN]: свойства материала заданные в узлах 
    # Элементные граничные условия
    d_elem_bc_types: np.ndarray,        # Index[EC]
    d_elem_bc_begin: np.ndarray,        # Float[EC]
    d_elem_bc_end: np.ndarray  ,        # Float[EC]
    d_elem_bc_steps: np.ndarray ,       # Index[EC]
    d_elem_bc_nodes: np.ndarray,        # Index[EC]
    d_elem_bc_time_interp: np.ndarray,  # Index[EC]
    d_elem_bc_index: np.ndarray,        # Index[2*ECI] (пары: ec_id, space_pos)
    d_elem_bc_index_offsets: np.ndarray,# Index[EN+1]
    d_elem_bc_data: np.ndarray,         # Float[ECD]
    d_elem_bc_data_offsets: np.ndarray, # Index[EC+1]
    current_time: Float,                # Float: текущее время
):
    """
    Элементное ядро вычисления внутренних упругих сил.
    
    Алгоритм (для каждого узла элемента параллельно):
    1. Загружает перемещения всех узлов элемента в shared memory
    2. Вычисляет градиент перемещений ∇u в локальных координатах
    3. Трансформирует в физические координаты через якобиан: ∇u_physical = ∇u_local × J^{-1}
    4. Вычисляет тензор напряжений σ через закон Гука (плоская деформация)
    5. Вычисляет θ = V × J^{-T} × σ (промежуточный тензор для интегрирования)
    6. Вычисляет дивергенцию div(θ) для получения внутренней силы
    7. Трансформирует силу в глобальные координаты
    8. Атомарно добавляет силу в глобальный массив d_nodes_sub_state
    
    Реализовано элементное ГУ:
    - Тип 0: Объемное давление (скаляр). После расчёта σ добавляется к диагоналям σ_xx и σ_yy.
    """

    """Вычисляем нашу позицию"""

    eid = cuda.blockIdx.x

    if eid >= e_count:
        return 

    nodes_in_element = d_elem_nodes_offsets[eid+1]-d_elem_nodes_offsets[eid]

    nu = cuda.threadIdx.x

    if nu >= nodes_in_element:
        return 

    enid = d_elem_nodes_offsets[eid]+nu
    nid = d_elem_nodes[enid]

    # print(f'< {eid}:{nu}({enid} -> {nid})> \n')

    """ Элементные ГУ: объемное давление (тип=0), 1D """

    bc_index_offset = d_elem_bc_index_offsets[enid]
    bcs_count = d_elem_bc_index_offsets[enid+1] - bc_index_offset

    bc_time_indexes = cuda.local.array((4), dtype=Index)
    empty_array(bc_time_indexes)

    bc_time_weights = cuda.local.array((4), dtype=Float)
    empty_array(bc_time_weights)

    added_pressure = 0.0
    for i in range(bcs_count):
        ec_id = d_elem_bc_index[0, bc_index_offset + i]
        ec_pos = d_elem_bc_index[1, bc_index_offset + i]

        ec_type = d_elem_bc_types[ec_id]

        # Временная активность
        time_begin = d_elem_bc_begin[ec_id]
        time_end = d_elem_bc_end[ec_id]
        time_step = d_elem_bc_steps[ec_id]

        if time_step <= 0 or current_time < time_begin or current_time > time_end:
            continue

        time_interp = d_elem_bc_time_interp[ec_id]
        bc_points = get_time_interpol_bspline(
            time_begin, time_end, time_step, time_interp, current_time,
            bc_time_indexes, bc_time_weights
        )

        ec_nodes = d_elem_bc_nodes[ec_id]
        data_offset = d_elem_bc_data_offsets[ec_id]

        # 1D условие: одна компонента
        value = 0.0
        for k in range(bc_points):
            value += d_elem_bc_data[data_offset + ec_nodes*bc_time_indexes[k] + ec_pos] * bc_time_weights[k]

        # Тип 0: объемное давление → добавить к диагоналям σ
        if ec_type == 0:
            added_pressure += value

    """ Расшарим перемещения """

    s_nodes_displacement = cuda.shared.array((DIM, THREADS_COUNT), Float)

    for d in range(DIM):
        s_nodes_displacement[d, nu] = d_nodes_state[d, nid]

    theta = cuda.shared.array((2,2,THREADS_COUNT), Float)

    cuda.syncthreads() 

    """ Вычислим градиент перемещения """

    volume = d_en_volumes[enid]

    yacoby = cuda.local.array((2, 2), dtype=np.float64)
    empty_array(yacoby)

    for i in range(2):
        for j in range(2):
            yacoby[i,j] = d_en_yacobies[i,j,enid]

    gradU = cuda.local.array((2, 2), dtype=np.float64)
    empty_array(gradU)

    local_grad(
        eid, nu, 
        s_nodes_displacement, DIM,         
        d_ept_nodes, 
        d_ept_dims,
        d_ept_nabla,
        d_ept_nabla_offsets,
        d_elem_types,
        d_elem_types_offsets,
        gradU
    )

    matmul_inplace(gradU, yacoby, 2, 2, 2)

    """ """

    lame_la = d_mat_props[1, enid] 
    lame_mu = d_mat_props[2, enid] 

    # Напряжения (плоская деформация)
    sigma_xx = lame_la * (gradU[0,0] + gradU[1,1]) + 2 * lame_mu * gradU[0,0] + added_pressure
    sigma_yy = lame_la * (gradU[0,0] + gradU[1,1]) + 2 * lame_mu * gradU[1,1] + added_pressure
    sigma_xy = lame_mu * (gradU[0,1] + gradU[1,0])


    # Используем J^{-T}: theta = V * J^{-T} * sigma
    theta[0,0,nu] = volume*(yacoby[0, 0]*sigma_xx + yacoby[1, 0]*sigma_xy)
    theta[0,1,nu] = volume*(yacoby[0, 0]*sigma_xy + yacoby[1, 0]*sigma_yy)
    theta[1,0,nu] = volume*(yacoby[0, 1]*sigma_xx + yacoby[1, 1]*sigma_xy)
    theta[1,1,nu] = volume*(yacoby[0, 1]*sigma_xy + yacoby[1, 1]*sigma_yy)

    cuda.syncthreads() 

    force = cuda.local.array((1,2), dtype=np.float64)
    empty_array(force)

    local_div(
        eid, nu, 
        theta, DIM,         
        d_ept_nodes, 
        d_ept_dims,
        d_ept_nabla,
        d_ept_nabla_offsets,
        d_elem_types,
        d_elem_types_offsets,
        force
    )

    # matmul_inplace(force, yacoby, 1, 2, 2)
    if force[0,0] != 0 or force[0,1] != 0:
        pass

    # # Записываем детерминированно в буфер EN (один поток -> один enid)
    # d_en_internal_forces[0, enid] = force[0,0]
    # d_en_internal_forces[1, enid] = force[0,1]


    # if (nid == 96):
    #     print('ify', current_time, nid, enid, force[0,1], d_nodes_sub_state[1,nid])

    cuda.atomic.add(d_nodes_sub_state, (0, nid), force[0,0])
    cuda.atomic.add(d_nodes_sub_state, (1, nid), force[0,1])




@cuda.jit
def mass_cdamping_element_kernel(
    d_mat_props: np.ndarray, # float[MP,EN]: свойства материала заданные в узлах 
    d_en_volumes: np.ndarray, # float[EN]: интегралы функций форм узлах элементов.
    d_elem_nodes: np.ndarray, # int[EN>E]: узлы элементов
    d_elem_nodes_offsets: np.ndarray, # int[E+1]: оффсеты узлов элементов
    e_count: Index, # int: число элементов
    d_mass: np.ndarray, # float[N]: матрица масс 
    d_damping: np.ndarray, # float[2,N]: анизотропные материалы 
):
    
    """Вычисляем нашу позицию"""

    eid = cuda.blockIdx.x

    if eid >= e_count:
        return 

    nodes_in_element = d_elem_nodes_offsets[eid+1]-d_elem_nodes_offsets[eid]

    nu = cuda.threadIdx.x

    if nu >= nodes_in_element:
        return 

    enid = d_elem_nodes_offsets[eid]+nu
    nid = d_elem_nodes[enid]

    density = d_mat_props[0,enid]
    alpha_damping_x = d_mat_props[3,enid]
    alpha_damping_y = d_mat_props[4,enid]

    cuda.atomic.add(d_mass, (nid), d_en_volumes[enid]*density)
    cuda.atomic.add(d_damping, (0,nid), d_en_volumes[enid]*density*alpha_damping_x)
    cuda.atomic.add(d_damping, (1,nid), d_en_volumes[enid]*density*alpha_damping_y)



@cuda.jit
def newmark_node_kernel(
    n_count,
    d_nodes_state,
    d_nodes_sub_state,

    d_mass: np.ndarray, # float[N]: матрица масс 
    d_damping: np.ndarray, # float[2,N]: анизотропные материалы 

    d_node_bc_types: np.ndarray,        # Index[NC]
    d_node_bc_begin: np.ndarray,        # Float[NC]
    d_node_bc_end: np.ndarray  ,        # Float[NC]
    d_node_bc_steps: np.ndarray ,       # Index[NC]
    d_node_bc_nodes: np.ndarray,        # Index[NC]
    d_node_bc_time_interp: np.ndarray,  # Index[NC] (0=step,1=linear)
    d_node_bc_index: np.ndarray,        # Float[2*NCI] (SoA: nc_id..., space_pos...)
    d_node_bc_index_offsets: np.ndarray,  # Index[N+1]
    d_node_bc_data: np.ndarray,         # Float[NCD]
    d_node_bc_data_offsets: np.ndarray, # Index[NC+1]
    current_time,
    dt,
):
    """
    Узловое ядро интегрирования по времени (схема Ньюмарка).
    
    Реализует явную схему Ньюмарка с β=0, γ=1/2 (центрально-разностная схема):
    
    1. Вычисление ускорений:
       a^n = (M + Δt/2·C)^{-1} × (f_ext^n - f_int^n - C·v^{n-1/2})
       
       где:
       - M - диагональная матрица масс
       - C - диагональная матрица демпфирования Релея (только массовая компонента α)
       - f_ext - внешние силы (из узловых ГУ)
       - f_int - внутренние упругие силы (из элементного ядра)
       - v^{n-1/2} - скорости на предыдущем полушаге
    
    2. Обновление скоростей:
       v^{n+1/2} = v^{n-1/2} + Δt·a^n
    
    3. Обновление перемещений:
       u^{n+1} = u^n + Δt·v^{n+1/2}
    
    4. Применение узловых граничных условий:
       - Закрепление DOF (Dirichlet)
       - Заданные скорости
       - Добавка к ускорению (внешние силы, уже поделенные на массу)
    
    Реализованы типы узловых ГУ:
    - Тип 0: внешняя сила по X
    - Тип 1: внешняя сила по Y
    - Тип 2: Дирихле — перемещение по X
    - Тип 3: Дирихле — перемещение по Y
    - Тип 4: Дирихле — скорость по X
    - Тип 5: Дирихле — скорость по Y
    """

    nid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if nid >= n_count:
        return

    # === ШАГ 1: Сбор внешних сил из узловых граничных условий ===
    
    external_force_x = 0.0
    external_force_y = 0.0

    # Флаги/значения для условий Дирихле
    clamp_u_x = False
    clamp_u_y = False
    clamp_v_x = False
    clamp_v_y = False
    clamp_u_x_value = 0.0
    clamp_u_y_value = 0.0
    clamp_v_x_value = 0.0
    clamp_v_y_value = 0.0

    # Получаем список граничных условий для данного узла
    node_bc_index_offset = d_node_bc_index_offsets[nid]
    bcs_count = d_node_bc_index_offsets[nid+1] - node_bc_index_offset

    # Буферы для B-сплайн интерполяции по времени (до кубической степени)
    bc_time_indexes = cuda.local.array((4), dtype=Index)
    empty_array(bc_time_indexes)

    bc_time_weights = cuda.local.array((4), dtype=Float)
    empty_array(bc_time_weights)

    # Проходим по всем ГУ данного узла
    for i in range(bcs_count):

        # Читаем индекс ГУ и позицию узла в данных ГУ
        nc_id = d_node_bc_index[0, node_bc_index_offset + i]
        nc_pos = d_node_bc_index[1, node_bc_index_offset + i]

        nc_type = d_node_bc_types[nc_id]

        # Проверяем, активно ли ГУ в текущее время
        time_begin = d_node_bc_begin[nc_id]
        time_end = d_node_bc_end[nc_id]
        time_step = d_node_bc_steps[nc_id]

        if time_step <= 0 or current_time < time_begin or current_time > time_end:
            continue  # ГУ неактивно в данное время

        bc_data_offsets = d_node_bc_data_offsets[nc_id]
        time_interp = d_node_bc_time_interp[nc_id]

        # Получаем индексы и веса для B-сплайн интерполяции по времени
        # Возвращает число точек (1-4 в зависимости от степени сплайна)
        bc_points = get_time_interpol_bspline(
            time_begin, time_end, time_step, time_interp, current_time, 
            bc_time_indexes, bc_time_weights
        )

        bc_nodes = d_node_bc_nodes[nc_id]

        # Интерполируем значение ГУ по времени
        value = 0.0
        for k in range(bc_points):
            value += d_node_bc_data[bc_data_offsets + bc_nodes*bc_time_indexes[k] + nc_pos] * bc_time_weights[k]

        # Применяем ГУ в зависимости от типа
        # Тип 0: внешняя сила по X
        # Тип 1: внешняя сила по Y
        # (Типы Dirichlet будут добавлены позже)
        if nc_type == 0:
            external_force_x += value
        elif nc_type == 1:
            external_force_y += value
        elif nc_type == 2:
            clamp_u_x = True
            clamp_u_x_value = value
        elif nc_type == 3:
            clamp_u_y = True
            clamp_u_y_value = value
        elif nc_type == 4:
            clamp_v_x = True
            clamp_v_x_value = value
        elif nc_type == 5:
            clamp_v_y = True
            clamp_v_y_value = value


    # === ШАГ 2: Вычисление ускорений по формуле Ньюмарка ===
    
    # Читаем предрассчитанные диагональные массу и демпфирование
    mass = d_mass[nid]
    damping_x = d_damping[0,nid]  # Анизотропное демпфирование по X
    damping_y = d_damping[1,nid]  # Анизотропное демпфирование по Y

    # Читаем внутренние упругие силы (из элементного ядра)
    internal_force_x = d_nodes_sub_state[0,nid]
    internal_force_y = d_nodes_sub_state[1,nid]

    # Вычисляем эффективную обратную массу: (M + Δt/2·C)^{-1}
    # Это диагональная матрица, поэтому просто инвертируем скаляры
    if mass  == 0:
       pass 

    inmass_x = 1.0 / (mass + damping_x*dt/2.0)
    inmass_y = 1.0 / (mass + damping_y*dt/2.0)

    # Читаем текущие перемещения
    displacement_x = d_nodes_state[0,nid]
    displacement_y = d_nodes_state[1,nid]

    # Читаем текущие скорости v^{n-1/2}
    velocity_x = d_nodes_state[2,nid]
    velocity_y = d_nodes_state[3,nid]

    # Вычисляем ускорения:
    # a^n = (M + Δt/2·C)^{-1} × (f_ext - f_int - C·v^{n-1/2})
    acceleration_x = (external_force_x - internal_force_x - damping_x*velocity_x) * inmass_x
    acceleration_y = (external_force_y - internal_force_y - damping_y*velocity_y) * inmass_y

    # === ШАГ 3: Обновление скоростей ===
    # v^{n+1/2} = v^{n-1/2} + Δt·a^n
    
    velocity_x += dt*acceleration_x
    velocity_y += dt*acceleration_y

    # === ШАГ 4: Обновление перемещений ===
    # u^{n+1} = u^n + Δt·v^{n+1/2}

    # === ШАГ 4.1: Применение условий Дирихле (жесткая фиксация) ===

    if clamp_v_x:
        velocity_x = clamp_v_x_value
    if clamp_v_y:
        velocity_y = clamp_v_y_value

    if clamp_u_x:
        velocity_x = (clamp_u_x_value - displacement_x)/dt
    if clamp_u_y:
        velocity_y = (clamp_u_y_value - displacement_y)/dt

    # === ШАГ 5: Записываем новые скорости и перемещения ===

    d_nodes_state[0,nid] = displacement_x + dt*velocity_x
    d_nodes_state[1,nid] = displacement_y + dt*velocity_y
    d_nodes_state[2,nid] = velocity_x
    d_nodes_state[3,nid] = velocity_y


@cuda.jit
def receiver_kernel(
    d_nodes_state,        # Float[ST, N]: состояние узлов (u_x, u_y, v_x, v_y)
    d_nodes_sub_state,    # Float[dim, N]: вспомогательное состояние (a_x, a_y)
    d_recv_components,    # Index[RC]: индексы компонент для записи
    d_recv_components_offsets,  # Index[R+1]: оффсеты компонент
    d_recv_index,         # Index[2, RI]: пары (recv_id, space_pos)
    d_recv_index_offsets, # Index[N+1]: оффсеты для каждого узла
    d_recv_data,          # Float[RD]: буфер одного кадра (OUTPUT)
    d_recv_data_offsets,  # Index[R+1]: оффсеты данных (RD = components * nodes_in_receiver)
    dim,                  # Index: размерность задачи
    n,                    # Index: число узлов
):
    """
    Ядро вывода (приёмники).
    Пишет «снимок» состояния в буфер одного кадра.
    Формат ReceiverData: SoA-порядок [компонента → узлы]
    Компоненты: [u_x, u_y, v_x, v_y, a_x, a_y] (индексы 0-5 для DIM=2)
    """
    nid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if nid >= n:
        return
    
    # Получаем список приемников для данного узла
    recv_index_offset = d_recv_index_offsets[nid]
    recv_count = d_recv_index_offsets[nid + 1] - recv_index_offset
    
    if recv_count == 0:
        return  # Для данного узла нет приемников
    
    # Проходим по всем приемникам этого узла (обычно один)
    for i in range(recv_count):
        # Читаем (recv_id, space_pos) из индекса
        recv_id = d_recv_index[0, recv_index_offset + i]
        space_pos = d_recv_index[1, recv_index_offset + i]

        # Получаем список компонент для записи
        components_offset = d_recv_components_offsets[recv_id]
        n_components = d_recv_components_offsets[recv_id + 1] - components_offset
        
        data_offset = d_recv_data_offsets[recv_id]
        data_size = d_recv_data_offsets[recv_id + 1] - data_offset
        # data_size = n_components * n_nodes_in_receiver
        if n_components == 0 or data_size == 0:
            continue
        
        n_nodes_in_receiver = data_size // n_components
        
        # Собираем массив полного состояния: [u_x, u_y, v_x, v_y, a_x, a_y]
        # Для DIM=2: ST=4 (u_x, u_y, v_x, v_y), SST=2 (a_x, a_y)
        full_state = cuda.local.array(6, dtype=np.float64)
        empty_array(full_state)

        # Заполняем из основного состояния (u, v)
        for d in range(4):  # ST=4
            full_state[d] = d_nodes_state[d, nid]
        
        # Заполняем из вспомогательного состояния (a)
        for d in range(dim):  # dim=2
            full_state[4 + d] = d_nodes_sub_state[d, nid]
        
        # Записываем запрошенные компоненты
        # SoA-порядок: [компонента → узлы]
        for comp_idx in range(n_components):
            component_id = d_recv_components[components_offset + comp_idx]
            
            # Проверяем корректность индекса компоненты
            if component_id < 0 or component_id >= 6:  # 4 + 2 = 6 компонент всего
                continue
            
            # Вычисляем позицию в массиве данных
            # data[component_idx][space_pos]
            pos = data_offset + comp_idx * n_nodes_in_receiver + space_pos
            
            # Записываем значение
            d_recv_data[pos] = full_state[component_id]



def solver(si: SolverInterface, frame_handler=None, stop_event=None):
    """
    Главная функция GPU-решателя для задач динамической упругости.
    
    Реализует явную схему метода конечных элементов с интегрированием по времени
    по схеме Ньюмарка (β=0, γ=1/2).
    
    Алгоритм:
    1. Валидация входных данных
    2. Загрузка всех данных на GPU (device)
    3. Предрасчет геометрии: якобианы и объемы (веса квадратур)
    4. Предрасчет диагональных матриц масс и демпфирования
    5. Временной цикл:
       a) Элементное ядро: расчет внутренних упругих сил f_int = Ku
       b) Узловое ядро: интегрирование по времени (Ньюмарк)
       c) Ядро вывода: запись данных в приемники
    6. Копирование результатов обратно на хост
    
    Параметры:
    ----------
    si : SolverInterface
        Структура с полным описанием задачи (см. INTERFACE.md)
    frame_handler : Callable[[float, int, np.ndarray], None], optional
        Функция-обработчик кадров: handler(time, frame_idx, data_flat)
    stop_event : threading.Event, optional
        Event для остановки вычислений
    
    Возвращает:
    -----------
    None (изменяет si.state.nodes_state и si.receivers.receivers_data на месте)
    """

    if not cuda.is_available():
        raise RuntimeError("CUDA не доступна (numba.cuda.is_available() == False)")

    # ===== ЭТАП 1: Валидация и подготовка параметров =====

    validate_interface(si)

    dt = (si.time.time_end - si.time.time_begin) / si.time.time_steps
    total_steps = int(si.time.time_steps)
    report_cuda_mem("start")

    # ===== ЭТАП 2: Загрузка данных на GPU =====
    
    # Основное состояние узлов: перемещения u и скорости v^{n-1/2}
    # Формат: [ST, N] где ST=4 для 2D (u_x, u_y, v_x, v_y)
    d_nodes_state = cuda.to_device(si.state.nodes_state)
    
    # Вспомогательное состояние: внутренние упругие силы (заполняется элементным ядром)
    # Формат: [dim, n] для доступа как d_nodes_sub_state[d, nid]
    nodes_sub_state = np.zeros((si.geom.dim, si.geom.n), dtype=Float)
    d_nodes_sub_state = cuda.to_device(nodes_sub_state)
    report_cuda_mem("after nodes_state + nodes_sub_state")

    # Простые типы элементов (prime)
    d_ept_nodes = cuda.to_device(si.primes.ept_nodes)
    d_ept_dims = cuda.to_device(si.primes.ept_dims)
    d_ept_weights = cuda.to_device(si.primes.ept_weights)
    d_ept_weights_offsets = cuda.to_device(si.primes.ept_weights_offsets)
    d_ept_nabla = cuda.to_device(np.array(si.primes.ept_nabla_shapes, dtype=Float, copy=False))
    d_ept_nabla_offsets = cuda.to_device(np.array(si.primes.ept_nabla_shapes_offsets, copy=False))
    report_cuda_mem("after primes")

    # Составные элементы (composite)
    d_elements_types = cuda.to_device(np.array(si.elems.elements_types, copy=False))
    d_elements_types_offsets = cuda.to_device(np.array(si.elems.elements_types_offsets, copy=False))
    d_elements_nodes = cuda.to_device(np.array(si.elems.elements_nodes, copy=False))
    d_elements_nodes_offsets = cuda.to_device(np.array(si.elems.elements_nodes_offsets, copy=False))
    report_cuda_mem("after elements")

    # Материалы (SoA)
    d_material_props = cuda.to_device(np.array(si.materials.material_props, dtype=Float, copy=False))
    report_cuda_mem("after materials")

    # Узловые ГУ
    d_nodes_bc_types = cuda.to_device(np.array(si.node_bcs.nodes_bc_types, copy=False))
    d_nodes_bc_begin = cuda.to_device(np.array(si.node_bcs.nodes_bc_begin, dtype=Float, copy=False))
    d_nodes_bc_end = cuda.to_device(np.array(si.node_bcs.nodes_bc_end, dtype=Float, copy=False))
    d_nodes_bc_steps = cuda.to_device(np.array(si.node_bcs.nodes_bc_steps, copy=False))
    d_nodes_bc_nodes = cuda.to_device(np.array(si.node_bcs.nodes_bc_nodes, copy=False))
    d_nodes_bc_time_interp = cuda.to_device(np.array(si.node_bcs.nodes_bc_time_interp_type, copy=False))
    d_nodes_bc_index = cuda.to_device(np.array(si.node_bcs.nodes_bc_index, copy=False))
    d_nodes_bc_index_offsets = cuda.to_device(np.array(si.node_bcs.nodes_bc_index_offsets, copy=False))
    d_nodes_bc_data = cuda.to_device(np.array(si.node_bcs.nodes_bc_data, dtype=Float, copy=False))
    d_nodes_bc_data_offsets = cuda.to_device(np.array(si.node_bcs.nodes_bc_data_offsets, copy=False))
    report_cuda_mem("after node BCs")

    # Элементные ГУ
    d_elems_bc_types = cuda.to_device(np.array(si.elem_bcs.elems_condition_types, copy=False))
    d_elems_bc_begin = cuda.to_device(np.array(si.elem_bcs.elems_condition_begin, dtype=Float, copy=False))
    d_elems_bc_end = cuda.to_device(np.array(si.elem_bcs.elems_condition_end, dtype=Float, copy=False))
    d_elems_bc_steps = cuda.to_device(np.array(si.elem_bcs.elems_condition_steps, copy=False))
    d_elems_bc_nodes = cuda.to_device(np.array(si.elem_bcs.elems_condition_nodes, copy=False))
    d_elems_bc_time_interp = cuda.to_device(np.array(si.elem_bcs.elems_condition_time_interp_type, copy=False))
    d_elems_bc_index = cuda.to_device(np.array(si.elem_bcs.elems_condition_index, copy=False))
    d_elems_bc_index_offsets = cuda.to_device(np.array(si.elem_bcs.elems_condition_index_offsets, copy=False))
    d_elems_bc_data = cuda.to_device(np.array(si.elem_bcs.elems_condition_data, dtype=Float, copy=False))
    d_elems_bc_data_offsets = cuda.to_device(np.array(si.elem_bcs.elems_condition_data_offsets, copy=False))
    report_cuda_mem("after elem BCs")

    # Приемники (output): буфер одного кадра
    d_recv_components = cuda.to_device(np.array(si.receivers.receivers_components, copy=False))
    d_recv_components_offsets = cuda.to_device(np.array(si.receivers.receivers_components_offsets, copy=False))
    d_recv_index = cuda.to_device(np.array(si.receivers.receivers_index, copy=False))
    d_recv_index_offsets = cuda.to_device(np.array(si.receivers.receivers_index_offsets, copy=False))
    # Оценим память буфера одного кадра
    rd = int(si.receivers.receivers_data.size)  # должен быть components * nodes_in_receiver
    item_size = np.dtype(Float).itemsize
    recv_bytes = rd * item_size
    print(f"[Receivers] frame_components={int(si.receivers.receivers_components_offsets[1] - si.receivers.receivers_components_offsets[0])}, "
          f"frame_entries={rd}, approx_size={recv_bytes/1024/1024:.2f} MB (dtype={np.dtype(Float)})")
    d_recv_data = cuda.to_device(np.array(si.receivers.receivers_data, dtype=Float, copy=False))
    d_recv_data_offsets = cuda.to_device(np.array(si.receivers.receivers_data_offsets, copy=False))
    
    # Pinned memory для эффективной передачи данных с GPU
    pinned_recv_buffer = cuda.pinned_array(rd, dtype=Float) if frame_handler is not None else None
    
    report_cuda_mem("after receivers")

    # ===== ЭТАП 3: Предрасчет геометрии =====
    
    # Координаты узлов в формате [dim, n] (SoA)
    d_nodes_coords = cuda.to_device(si.geom.nodes_coords)
    
    # Создаем массивы для предрасчитанной геометрии
    en = int(si.elems.en)  # Суммарное число узлов во всех элементах (EN)
    dim = int(si.geom.dim)
    


    # Объемы (веса квадратур): V_q = w_q × |J(ξ_q,η_q)|
    en_volumes = np.zeros(en, dtype=Float)
    d_en_volumes = cuda.to_device(en_volumes)
    
    # Обратные якобианы: J^{-1} для трансформации градиентов
    # Формат: [dim, dim, en]
    en_yacobies = np.zeros((dim, dim, en), dtype=Float)
    d_en_yacobies = cuda.to_device(en_yacobies)
    report_cuda_mem("after geometry alloc")
    
    # Конфигурация запуска элементных ядер
    # Стратегия: один блок на элемент, потоки внутри блока - узлы элемента
    elements_count = int(si.elems.e)
    # Число узлов в элементе берём из offsets (максимум по всем элементам)

    nodes_per_element = int(np.max(si.elems.elements_nodes_offsets[1:] - si.elems.elements_nodes_offsets[:-1]))
    threads_per_element = min(nodes_per_element, THREADS_COUNT)
    nodes_count = int(si.geom.n)
    # Запускаем ядро инициализации геометрии
    # Для каждого узла каждого элемента вычисляет:
    # - Обратный якобиан J^{-1} для трансформации из локальных координат в физические
    # - Объем V = w × |J| (произведение весов квадратуры на детерминант якобиана)
    init_element_kernel[elements_count, threads_per_element](
        elements_count,
        en,
        d_elements_nodes,
        d_elements_nodes_offsets,
        d_nodes_coords,
        d_ept_nodes,
        d_ept_dims,
        d_ept_nabla,
        d_ept_nabla_offsets,
        d_elements_types,
        d_elements_types_offsets,
        d_ept_weights,
        d_ept_weights_offsets,
        d_en_volumes,
        d_en_yacobies,
    )
    report_cuda_mem("after geometry init")
    
    # ===== ЭТАП 4: Предрасчет диагональных матриц масс и демпфирования =====
    
    # Создаем массивы для сборки диагональных матриц
    mass = np.zeros(nodes_count, dtype=Float)
    d_mass = cuda.to_device(mass)
    
    # Демпфирование анизотропное (разное по X и Y)
    damping = np.zeros((dim, nodes_count), dtype=Float)
    d_damping = cuda.to_device(damping)
    report_cuda_mem("after mass+damping alloc")
    
    # Запускаем ядро сборки масс и демпфирования
    # Для каждого узла элемента атомарно добавляет вклад в глобальные массу и демпфирование:
    # M[node] += ρ × V
    # C[node] += ρ × α × V
    # где ρ - плотность, α - коэффициент Релея, V - объем (вес квадратуры)
    mass_cdamping_element_kernel[elements_count, threads_per_element](
        d_material_props,
        d_en_volumes,
        d_elements_nodes,
        d_elements_nodes_offsets,
        elements_count,
        d_mass,
        d_damping,
    )
    report_cuda_mem("after mass+damping build")

    # Профилирование только аллокаций (без шага по времени)
    if os.getenv("GPU_PROFILE_ALLOC_ONLY", "0") == "1":
        print("[GPU Solver] GPU_PROFILE_ALLOC_ONLY=1 → завершаю после аллокаций.")
        return

    # ===== ЭТАП 5: Временной цикл интегрирования (схема Ньюмарка) =====
    
    # Конфигурация для узловых ядер: один поток на узел
    threads_per_block_node = THREADS_COUNT
    blocks_per_grid_node = (nodes_count + threads_per_block_node - 1) // threads_per_block_node
    
    # Инициализация времени
    t = float(si.time.time_begin)
    
    # Подготовка логики приёмников: R ожидается 1
    r = int(si.receivers.r) if hasattr(si.receivers, "r") else 1
    recv_begin = float(si.receivers.receivers_begin[0]) if r > 0 else 0.0
    recv_end = float(si.receivers.receivers_end[0]) if r > 0 else 0.0
    frames_count = int(si.receivers.receivers_steps[0]) if r > 0 else 0  # steps трактуем как число кадров
    has_receiver = (r > 0 and frames_count > 0 and recv_end >= recv_begin)
    
    # Вычисляем, через сколько шагов вызывать handler (равномерное разбиение)
    steps_per_frame = max(1, total_steps // frames_count) if frames_count > 0 else total_steps
    next_frame_idx = 0

    # Цикл по временным шагам
    for step in range(total_steps):
        # Проверка stop_event
        if stop_event is not None and stop_event.is_set():
            print(f"[GPU Solver] Остановка по stop_event на шаге {step}/{total_steps}")
            break
        
        # ===== ШАГ 5.1: Расчет внутренних упругих сил (Элементное ядро) =====
        
        empty_array_kernel[blocks_per_grid_node, THREADS_COUNT](d_nodes_sub_state, 2, nodes_count)
        # Обнуляем буфер EN сил

        # Запускаем элементное ядро
        # Для каждого узла элемента параллельно:
        # 1. Вычисляет градиент перемещений ∇u
        # 2. Вычисляет напряжения σ = D·ε через закон Гука
        # 3. Вычисляет внутренние силы f_int = ∫ B^T·σ dΩ
        # 4. Атомарно добавляет в d_nodes_sub_state
        inner_forces_element_kernel[elements_count, threads_per_element](
            d_ept_nodes,
            d_ept_dims,
            d_ept_nabla,
            d_ept_nabla_offsets,
            elements_count,
            d_elements_types,
            d_elements_types_offsets,
            d_elements_nodes,
            d_elements_nodes_offsets,
            d_en_volumes,
            d_en_yacobies,
            d_nodes_state,
            d_nodes_sub_state,
            d_material_props,
            d_elems_bc_types,
            d_elems_bc_begin,
            d_elems_bc_end,
            d_elems_bc_steps,
            d_elems_bc_nodes,
            d_elems_bc_time_interp,
            d_elems_bc_index,
            d_elems_bc_index_offsets,
            d_elems_bc_data,
            d_elems_bc_data_offsets,
            t,
        )
        
        # Сборка EN -> узлы детерминированно (без атомиков)
        # gather_en_forces_to_nodes_kernel[blocks_per_grid_node, THREADS_COUNT](
        #     t,
        #     d_en_internal_forces,
        #     d_nodes_sub_state,
        #     d_node_en_index,
        #     d_node_en_offsets,
        #     nodes_count,
        # )
        
        # ===== ШАГ 5.2: Интегрирование по времени (Узловое ядро) =====
        
        # Запускаем узловое ядро Ньюмарка
        # Для каждого узла параллельно:
        # 1. Собирает внешние силы из узловых ГУ (с интерполяцией по времени)
        # 2. Вычисляет ускорения: a = (M + Δt/2·C)^{-1}·(f_ext - f_int - C·v)
        # 3. Обновляет скорости: v^{n+1/2} = v^{n-1/2} + Δt·a^n
        # 4. Обновляет перемещения: u^{n+1} = u^n + Δt·v^{n+1/2}
        # 5. Применяет узловые ГУ (Dirichlet - в будущих версиях)
        newmark_node_kernel[blocks_per_grid_node, THREADS_COUNT](
            nodes_count,
            d_nodes_state,
            d_nodes_sub_state,
            d_mass,
            d_damping,
            d_nodes_bc_types,
            d_nodes_bc_begin,
            d_nodes_bc_end,
            d_nodes_bc_steps,
            d_nodes_bc_nodes,
            d_nodes_bc_time_interp,
            d_nodes_bc_index,
            d_nodes_bc_index_offsets,
            d_nodes_bc_data,
            d_nodes_bc_data_offsets,
            t,
            dt,
        )
        
        """</Шаг расчета скорости и перемещения>"""

        # ===== ШАГ 5.3: Запись данных в приемники и вызов handler =====
        
        # Вызываем handler периодически (каждые steps_per_frame шагов)
        if frame_handler is not None and (step + 1) % steps_per_frame == 0 and next_frame_idx < frames_count:
            # Пишем снимок текущего состояния в буфер одного кадра
            if has_receiver:
                receiver_kernel[blocks_per_grid_node, THREADS_COUNT](
                    d_nodes_state,
                    d_nodes_sub_state,
                    d_recv_components,
                    d_recv_components_offsets,
                    d_recv_index,
                    d_recv_index_offsets,
                    d_recv_data,
                    d_recv_data_offsets,
                    dim,
                    nodes_count,
                )
            
            # Синхронизируем GPU
            cuda.synchronize()
            
            # Копируем в pinned memory
            if pinned_recv_buffer is not None:
                d_recv_data.copy_to_host(pinned_recv_buffer)
                
                # Вызов handler на хосте
                try:
                    frame_handler(t, next_frame_idx, pinned_recv_buffer)
                except Exception as e:
                    print(f"[GPU Solver] Ошибка в frame_handler: {e}")
                
                next_frame_idx += 1
        
        # Продвигаем время на один шаг
        t += dt
        
        # Вывод прогресса (реже, чем каждый шаг)
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(f"[GPU Solver] Шаг {step + 1}/{total_steps}, t={t:.6f}")

    # ===== ЭТАП 6: Копирование результатов обратно на хост =====
    
    # Копируем обновленное состояние узлов (финальные u и v)
    si.state.nodes_state = d_nodes_state.copy_to_host()
    
    # Копируем данные приемников (временные ряды для постобработки)
    si.receivers.receivers_data = d_recv_data.copy_to_host()


