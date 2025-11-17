"""
Вспомогательные device-функции и утилиты для GPU-ядер.

Содержит функции, выполняющиеся на устройстве (device) и используемые
основными CUDA-ядрами для решения задач динамической упругости.
"""

from __future__ import annotations

import numpy as np
from numba import cuda
import math

from interface import Index, Float


# ===== Вспомогательные ядра =====



@cuda.jit
def empty_array_kernel(arr, dim_size, n_size):
    """
    Простое ядро для обнуления массива на устройстве.
    
    Используется для инициализации массивов в формате [dim, n] перед
    сборкой элементных вкладов атомарными операциями.
    
    Параметры:
    ----------
    arr : array[dim, n]
        Массив для обнуления
    dim_size : int
        Размерность первого индекса (обычно пространственная размерность)
    n_size : int
        Размерность второго индекса (обычно число узлов)
    """

    # Старый неверный код
    # idx = cuda.grid(1)
    # total_size = dim_size * n_size
    # if idx < total_size:
    #     d = idx // n_size
    #     nid = idx % n_size
    #     arr[d, nid] = 0.0

    nid = cuda.grid(1)

    if nid >= n_size:
        return

    for d in range(dim_size):
        arr[d, nid] = 0.0
 
def _format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for u in units:
        if size < 1024.0:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{size:.2f} PB"
 
def get_cuda_mem_info() -> tuple[int, int]:
    """
    Возвращает (free_bytes, total_bytes) по текущему CUDA-контексту.
    """
    ctx = cuda.current_context()
    free_b, total_b = ctx.get_memory_info()
    return int(free_b), int(total_b)
 
def report_cuda_mem(tag: str = "") -> None:
    """
    Печатает текущее состояние памяти CUDA.
    """
    try:
        free_b, total_b = get_cuda_mem_info()
        used_b = total_b - free_b
        prefix = f"[CUDA MEM]{' ' + tag if tag else ''}"
        print(f"{prefix}: used={_format_bytes(used_b)} free={_format_bytes(free_b)} total={_format_bytes(total_b)}")
    except Exception as e:
        print(f"[CUDA MEM]{' ' + tag if tag else ''}: <unavailable> ({e})")
 
# ===== Device-функции =====

@cuda.jit(device=True)
def empty_array(array):
    """
    Заполняет переданный массив нулями (in-place) на устройстве (device function).
    Работает для любых размерностей, используя flat-итерацию (C-order).
    """
    total_size = 1
    for s in array.shape:
        total_size *= s
    for i in range(total_size):
        array.flat[i] = 0.0

@cuda.jit(device=True)
def matmul_inplace(A, B, m, k, n):
    """
    Умножает матрицу A на матрицу B справа: A := A × B
    
    Параметры:
    ----------
    A : array[m, k] (вход/выход)
        Левая матрица, результат записывается обратно в нее
    B : array[k, n] (вход)
        Правая матрица
    m, k, n : int
        Размерности матриц
    
    Примечание:
    -----------
    Использует локальный массив (local memory) для хранения промежуточных
    результатов, что необходимо для корректной перезаписи A.
    Ограничение: n <= 64 (размер локального массива).
    """
    # Временный массив для хранения одной строки результата
    # Размещается в быстрой локальной памяти потока
    row_result = cuda.local.array(64, dtype=np.float64)  # Максимальная ширина N <= 64
    
    for i in range(m):  # для каждой строки A
        # Инициализация строки нулями
        for j in range(n):
            row_result[j] = 0.0
        
        # Вычисляем элементы строки i результата: C[i,j] = Σ_p A[i,p]·B[p,j]
        for p in range(k):
            a_val = A[i, p]
            for j in range(n):
                row_result[j] += a_val * B[p, j]
        
        # Копируем вычисленную строку обратно в A
        for j in range(n):
            A[i, j] = row_result[j]


@cuda.jit(device=True)
def get_time_interpol_bspline(time_begin, time_end, time_step, time_interp, current_time,
                                    indexes, weights):
    """
    B-сплайн интерполяция по времени для граничных условий и приемников.
    
    Вычисляет индексы и веса для интерполяции значения в момент current_time
    по дискретным значениям, заданным на time_step точках в интервале [time_begin, time_end].
    
    Параметры:
    ----------
    time_begin : float
        Начало временного интервала
    time_end : float
        Конец временного интервала
    time_step : int
        Число дискретных точек времени
    time_interp : int
        Степень интерполяции (0=константа, 1=линейная, 2=квадратичная, 3=кубическая)
    current_time : float
        Текущее время, для которого вычисляется значение
    indexes : array[4] (выход)
        Индексы опорных точек (массив должен быть предварительно выделен)
    weights : array[4] (выход)
        Веса опорных точек (массив должен быть предварительно выделен)
    
    Возвращает:
    -----------
    k : int
        Число опорных точек (1-4 в зависимости от степени)
        
    Примеры:
    --------
    degree=0: кусочно-постоянная (ступенька), k=1
    degree=1: линейная интерполяция, k=2
    degree=2: квадратичная B-сплайн, k=3
    degree=3: кубическая B-сплайн, k=4
    """
    # Проверка границ
    if time_step <= 0 or current_time < time_begin or current_time > time_end:
        return 0

    # Ограничиваем степень интерполяции диапазоном 0..3
    degree = int(time_interp)
    if degree < 0:
        degree = 0
    if degree > 3:
        degree = 3

    # Частный случай: одна точка
    if time_step == 1:
        indexes[0] = 0
        weights[0] = 1.0
        return 1

    # Нормализуем время в диапазон [0, 1]
    T = time_end - time_begin
    if math.fabs(T) < 1e-18:
        indexes[0] = 0
        weights[0] = 1.0
        return 1
    u = (current_time - time_begin) / T
    if u < 0.0: u = 0.0
    if u > 1.0: u = 1.0

    # Позиция внутри массива контрольных точек [0, time_step-1]
    pos = u * (time_step - 1)
    i = int(math.floor(pos))
    s = pos - i  # дробная часть в [0, 1)

    # Граничная защита
    if i < 0: i = 0
    if i > time_step - 2: i = time_step - 2

    # Вычисляем индексы и веса в зависимости от степени интерполяции
    if degree == 0:
        # Степень 0: кусочно-постоянная (константа)
        k = 1
        indexes[0] = i
        weights[0] = 1.0

    elif degree == 1:
        # Степень 1: линейная интерполяция
        k = 2
        indexes[0] = i
        indexes[1] = i + 1
        weights[0] = 1.0 - s
        weights[1] = s

    elif degree == 2:
        # Степень 2: квадратичная B-сплайн
        k = 3
        indexes[0] = i - 1
        indexes[1] = i
        indexes[2] = i + 1
        # Корректировка границ
        if indexes[0] < 0: indexes[0] = 0
        if indexes[2] > time_step-1: indexes[2] = time_step-1
        # Базисные функции квадратичного B-сплайна
        weights[0] = 0.5*(1.0 - s)*(1.0 - s)
        weights[1] = 0.5*(2.0*s*(1.0 - s) + 1.0)
        weights[2] = 0.5*s*s

    else:  # degree == 3
        # Степень 3: кубическая B-сплайн
        k = 4
        indexes[0] = i - 1
        indexes[1] = i
        indexes[2] = i + 1
        indexes[3] = i + 2
        # Корректировка границ
        if indexes[0] < 0: indexes[0] = 0
        if indexes[3] > time_step-1: indexes[3] = time_step-1
        # Базисные функции кубического B-сплайна
        s2 = s*s
        s3 = s2*s
        weights[0] = (1 - s)**3 / 6.0
        weights[1] = (3*s3 - 6*s2 + 4) / 6.0
        weights[2] = (-3*s3 + 3*s2 + 3*s + 1) / 6.0
        weights[3] = s3 / 6.0

    return k


@cuda.jit(device=True)
def local_div(
    eid: Index, # Индекс элемента
    nu: Index, # Локальный номер узла, в котором считаем градиент

    value: np.ndarray, # float[space_dim, value_dim, elem_nodes]: вектор узовых значений поля, градиент которых считаем
    value_dim: Index, # int: размерность данных 
    d_ept_nodes: np.ndarray, # int[EPT]: число узлов в простых типах элементов 
    d_ept_dims: np.ndarray, # int[EPT]: размерности простых типов элементов 
    d_ept_nabla: np.ndarray, # float[EPNS, EPT]: градиенты форм простых типов элементов
    d_ept_nabla_offsets: np.ndarray, # int[EPT+1]: оффсеты градиентов форм простых типов элементов
    d_elem_types: np.ndarray, # int[ET]: составные типы элементов
    d_elem_types_offsets: np.ndarray, # int[E+1]: оффсеты типов элементов

    result: np.ndarray, # float[1,value_dim]: Массив для записи выходных данных
):
    """
    Оператор локального дивергента.
    
    Вычисляет пространственный дивергент векторного поля в узле элемента
    по известным узловым значениям поля.
    
    Для составных элементов использует тензорное произведение градиентов
    простых типов элементов.
    
    Формула: div(v) = Σ_i ∂v_i/∂x_i
    
    Параметры:
    ----------
    eid : Index
        Индекс элемента
    nu : Index
        Локальный номер узла в элементе
    value : array[space_dim, value_dim, elem_nodes]
        Узловые значения поля
    value_dim : Index
        Размерность данных
    result : array[1, value_dim]
        Массив для записи результата (должен быть обнулен снаружи)
    """

    elem_types_offset = d_elem_types_offsets[eid]
    t_types_in_element = d_elem_types_offsets[eid+1] - elem_types_offset

    mod_mul = 1
    dim_sum = 0

    for k_type_id in range(t_types_in_element):
        type_id = d_elem_types[elem_types_offset+k_type_id]

        ept_nabla_offset = d_ept_nabla_offsets[type_id]

        nodes_k = d_ept_nodes[type_id]
        dim_k = d_ept_dims[type_id]

        nu_k = (nu // mod_mul) % nodes_k

        for la_k in range(nodes_k):
            la = nu + (la_k - nu_k)*mod_mul

            for d_k in range(dim_k):
                nabla_d_k_nu_k_la_k = d_ept_nabla[ept_nabla_offset+
                    nodes_k*(nodes_k*d_k + nu_k) + la_k 
                ]

                for d_v in range(value_dim):
                    result[0,d_v] += nabla_d_k_nu_k_la_k*value[d_k+dim_sum,d_v,la]
        
        mod_mul *= nodes_k
        dim_sum += dim_k


@cuda.jit(device=True)
def local_grad(
    eid: Index, # Индекс элемента
    nu: Index, # Локальный номер узла, в котором считаем градиент

    value: np.ndarray, # float[value_dim, elem_nodes]: вектор узовых значений поля, градиент которых считаем
    value_dim: Index, # int: размерность данных 
    
    d_ept_nodes: np.ndarray, # int[EPT]: число узлов в простых типах элементов 
    d_ept_dims: np.ndarray, # int[EPT]: размерности простых типов элементов 
    d_ept_nabla: np.ndarray, # float[EPNS, EPT]: градиенты форм простых типов элементов
    d_ept_nabla_offsets: np.ndarray, # int[EPT+1]: оффсеты градиентов форм простых типов элементов
    d_elem_types: np.ndarray, # int[ET]: составные типы элементов
    d_elem_types_offsets: np.ndarray, # int[E+1]: оффсеты типов элементов

    result: np.ndarray, # float[value_dim*dim]: Массив для записи выходных данных
):
    """
    Оператор локального градиента.
    
    Вычисляет пространственный градиент поля в узле элемента
    по известным узловым значениям поля.
    
    Для составных элементов использует тензорное произведение градиентов
    простых типов элементов.
    
    Формула: ∇f = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_dim]
    
    Для векторного поля возвращает якобиан: J[i,j] = ∂f_i/∂x_j
    
    Параметры:
    ----------
    eid : Index
        Индекс элемента
    nu : Index
        Локальный номер узла в элементе
    value : array[value_dim, elem_nodes]
        Узловые значения поля
    value_dim : Index
        Размерность данных (число компонент поля)
    result : array[value_dim, dim]
        Массив для записи результата (должен быть обнулен снаружи)
    """

    elem_types_offset = d_elem_types_offsets[eid]
    t_types_in_element = d_elem_types_offsets[eid+1] - elem_types_offset

    mod_mul = 1
    dim_sum = 0

    for k_type_id in range(t_types_in_element):
        type_id = d_elem_types[elem_types_offset+k_type_id]

        ept_nabla_offset = d_ept_nabla_offsets[type_id]

        nodes_k = d_ept_nodes[type_id]
        dim_k = d_ept_dims[type_id]

        nu_k = (nu // mod_mul) % nodes_k

        for mu_k in range(nodes_k):
            mu = nu + (mu_k - nu_k)*mod_mul

            for d_k in range(dim_k):

                test_index = ept_nabla_offset + nodes_k*(nodes_k*d_k + mu_k) + nu_k 
                nabla_d_k_mu_k_nu_k = d_ept_nabla[test_index]

                for d_v in range(value_dim):
                    adding_value = value[d_v,mu]*nabla_d_k_mu_k_nu_k
                    result[d_v, d_k+dim_sum] += adding_value
                    
        mod_mul *= nodes_k
        dim_sum += dim_k

