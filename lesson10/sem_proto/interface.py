from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


# Типы по спецификации прототипа: float64 для чисел, int для индексов
Float = np.float64
Index = np.int64


@dataclass
class SolverTime:
    time_begin: Float  # время начала процесса
    time_end: Float  # время окончания процесса
    time_steps: Index  # количество шагов интегрирования


@dataclass
class NodesGeometryInput:
    dim: Index # размерность пространства, должна быть равна 2
    n: Index # число узлов сетки
    dof: Index # число степеней свободы, равен DIM × N
    nodes_coords: np.ndarray  # Float[DIM, N], 2D, SoA: компоненты в первом измерении (x, y, z)


@dataclass
class PrimeElementTypesInput:
    ept: Index
    ept_nodes: np.ndarray  # Index[EPT]
    ept_dims: np.ndarray   # Index[EPT]
    epw: Index
    ept_weights: np.ndarray  # Float[EPW]
    ept_weights_offsets: np.ndarray  # Index[EPT+1] 
    epns: Index
    ept_nabla_shapes: np.ndarray  # Float[EPNS]
    ept_nabla_shapes_offsets: np.ndarray  # Index[EPT+1]


@dataclass
class CommonElementsInput:
    e: Index

    et: Index
    elements_types: np.ndarray          # Index[ET]
    elements_types_offsets: np.ndarray  # Index[E+1]

    en: Index
    elements_nodes: np.ndarray          # Index[EN]
    elements_nodes_offsets: np.ndarray  # Index[E+1]


@dataclass
class NodesStateInput:
    st: Index  # размерность основного состояния
    nodes_state: np.ndarray  # Index[ST,N], SoA; v-компоненты трактуются как v^{n+1/2}


@dataclass
class MaterialsInput:
    m: Index
    material_props: np.ndarray  # Float[M,EN], SoA; [E, nu, rho, alpha]


@dataclass
class NodeBCsInput:
    nc: Index
    nodes_bc_types: np.ndarray        # Index[NC]
    nodes_bc_begin: np.ndarray        # Float[NC]
    nodes_bc_end: np.ndarray          # Float[NC]
    nodes_bc_steps: np.ndarray        # Index[NC]
    nodes_bc_nodes: np.ndarray        # Index[NC]
    nodes_bc_time_interp_type: np.ndarray  # Index[NC] (0=step,1=linear)

    nci: Index
    nodes_bc_index: np.ndarray        # Float[2,NCI] (SoA: nc_id..., space_pos...)
    nodes_bc_index_offsets: np.ndarray  # Index[N+1]

    ncd: Index
    nodes_bc_data: np.ndarray         # Float[NCD]
    nodes_bc_data_offsets: np.ndarray # Index[NC+1]


@dataclass
class ElemBCsInput:
    ec: Index
    elems_condition_types: np.ndarray        # Index[EC]
    elems_condition_begin: np.ndarray        # Float[EC]
    elems_condition_end: np.ndarray          # Float[EC]
    elems_condition_steps: np.ndarray        # Index[EC]
    elems_condition_nodes: np.ndarray        # Index[EC]
    elems_condition_time_interp_type: np.ndarray  # Index[EC]

    eci: Index
    elems_condition_index: np.ndarray        # Index[2,ECI] (SoA: ec_id..., space_pos...)
    elems_condition_index_offsets: np.ndarray  # Index[EN+1]

    ecd: Index
    elems_condition_data: np.ndarray         # Float[ECD]
    elems_condition_data_offsets: np.ndarray # Index[EC+1]


@dataclass
class ReceiversOutput:
    r: Index
    receivers_types: np.ndarray           # Index[S]
    receivers_begin: np.ndarray           # Float[S]
    receivers_end: np.ndarray             # Float[S]
    receivers_steps: np.ndarray           # Index[S]

    rc: Index
    receivers_components: np.ndarray      # Index[SC]
    receivers_components_offsets: np.ndarray  # Index[S+1]

    ri: Index
    receivers_index: np.ndarray           # Index[2,SI] (SoA: s_id..., space_pos...)
    receivers_index_offsets: np.ndarray   # Index[N+1]

    rd: Index
    receivers_data: np.ndarray            # Float[SD]
    receivers_data_offsets: np.ndarray    # Index[S+1]


@dataclass
class SolverInterface:
    time: SolverTime
    geom: NodesGeometryInput
    state: NodesStateInput
    primes: PrimeElementTypesInput
    elems: CommonElementsInput
    materials: MaterialsInput
    node_bcs: NodeBCsInput
    elem_bcs: ElemBCsInput
    receivers: ReceiversOutput


def _require_array(name: str, a: np.ndarray, dtype, shape):
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} должен быть numpy.ndarray")
    if a.dtype != dtype:
        raise TypeError(f"{name} должен иметь dtype {dtype}, получен {a.dtype}")
    if shape is not None:
        if type(shape) == int or type(shape) == Index:
            shape = (shape,)
        if tuple(a.shape) != tuple(shape):
            raise ValueError(f"{name}: ожидался размер {shape}, получен {a.shape}")


def validate_interface(si: SolverInterface) -> None:
    """
    Быстрая валидация форм и типов
    Предназначена для ранней отладки интерфейса; не выполняет полную проверку данных.
    """

    # Геометрия
    if si.geom.dim not in (1,2,3):
        raise ValueError("DIM должен быть 1d 2d или 3d!")
    if si.geom.dof != si.geom.n * si.geom.dim:
        raise ValueError("DOF должен равняться N * DIM")
    _require_array("NodesCoords", si.geom.nodes_coords, Float, (si.geom.dim, si.geom.n) )


    # Prime types
    _require_array("ElementPrimeTypeNodes", si.primes.ept_nodes, Index,  si.primes.ept)
    _require_array("ElementPrimeTypeDims",  si.primes.ept_dims, Index, si.primes.ept)
    _require_array("ElementPrimeTypeWeight", si.primes.ept_weights, Float, si.primes.epw)
    _require_array("ElementPrimeTypeWeightOffsets", si.primes.ept_weights_offsets, Index, si.primes.ept + 1)
    _require_array("ElementPrimeTypeNablaShapes", si.primes.ept_nabla_shapes, Float, si.primes.epns)
    _require_array("ElementPrimeTypeNablaShapesOffsets", si.primes.ept_nabla_shapes_offsets, Index, si.primes.ept + 1)

    # Элементы (составные)
    _require_array("ElementsTypes", si.elems.elements_types, Index, si.elems.et)
    _require_array("ElementsTypesOffsets", si.elems.elements_types_offsets, Index, si.elems.e + 1)
    _require_array("ElementsNodes", si.elems.elements_nodes, Index, si.elems.en)
    _require_array("ElementsNodesOffsets", si.elems.elements_nodes_offsets, Index, si.elems.e + 1)

    # Состояние
    _require_array("NodesState", si.state.nodes_state, Float, (si.state.st , si.geom.n))

    # Materials
    _require_array("MaterialProps", si.materials.material_props, Float, (si.materials.m , si.elems.en))

    # Node BCs
    _require_array("NodesConditionTypes", si.node_bcs.nodes_bc_types, Index, si.node_bcs.nc)
    _require_array("NodesConditionBegin", si.node_bcs.nodes_bc_begin, Float, si.node_bcs.nc)
    _require_array("NodesConditionEnd", si.node_bcs.nodes_bc_end, Float, si.node_bcs.nc)
    _require_array("NodesConditionSteps", si.node_bcs.nodes_bc_steps, Index, si.node_bcs.nc)
    _require_array("NodesConditionNodes", si.node_bcs.nodes_bc_nodes, Index, si.node_bcs.nc)
    _require_array("NodesConditionTimeInterpType", si.node_bcs.nodes_bc_time_interp_type, Index, si.node_bcs.nc)
    _require_array("NodesConditionIndex", si.node_bcs.nodes_bc_index, Index, (2, si.node_bcs.nci))
    _require_array("NodesConditionIndexOffsets", si.node_bcs.nodes_bc_index_offsets, Index, si.geom.n + 1)
    _require_array("NodesConditionData", si.node_bcs.nodes_bc_data, Float, si.node_bcs.ncd)
    _require_array("NodesConditionDataOffsets", si.node_bcs.nodes_bc_data_offsets, Index, si.node_bcs.nc + 1)

    # Elem BCs
    _require_array("ElemsConditionTypes", si.elem_bcs.elems_condition_types, Index, si.elem_bcs.ec)
    _require_array("ElemsConditionBegin", si.elem_bcs.elems_condition_begin, Float, si.elem_bcs.ec)
    _require_array("ElemsConditionEnd", si.elem_bcs.elems_condition_end, Float, si.elem_bcs.ec)
    _require_array("ElemsConditionSteps", si.elem_bcs.elems_condition_steps, Index, si.elem_bcs.ec)
    _require_array("ElemsConditionNodes", si.elem_bcs.elems_condition_nodes, Index, si.elem_bcs.ec)
    _require_array("ElemsConditionTimeInterpType", si.elem_bcs.elems_condition_time_interp_type, Index, si.elem_bcs.ec)
    _require_array("ElemsConditionIndex", si.elem_bcs.elems_condition_index, Index, (2, si.elem_bcs.eci))
    _require_array("ElemsConditionIndexOffsets", si.elem_bcs.elems_condition_index_offsets, Index, si.elems.en + 1)
    _require_array("ElemsConditionData", si.elem_bcs.elems_condition_data, Float, si.elem_bcs.ecd)
    _require_array("ElemsConditionDataOffsets", si.elem_bcs.elems_condition_data_offsets, Index, si.elem_bcs.ec + 1)

    # Приемники (output)
    _require_array("ReceiverTypes", si.receivers.receivers_types, Index, si.receivers.r)
    _require_array("ReceiverBegin", si.receivers.receivers_begin, Float, si.receivers.r)
    _require_array("ReceiverEnd", si.receivers.receivers_end, Float, si.receivers.r)
    _require_array("ReceiverSteps", si.receivers.receivers_steps, Index, si.receivers.r)
    _require_array("ReceiverComponents", si.receivers.receivers_components, Index, si.receivers.rc)
    _require_array("ReceiverComponentsOffsets", si.receivers.receivers_components_offsets, Index, si.receivers.r + 1)
    _require_array("ReceiverIndex", si.receivers.receivers_index, Index, (2,si.receivers.ri))
    _require_array("ReceiverIndexOffsets", si.receivers.receivers_index_offsets, Index, si.geom.n + 1)
    _require_array("ReceiverData", si.receivers.receivers_data, Float, si.receivers.rd)
    _require_array("ReceiverDataOffsets", si.receivers.receivers_data_offsets, Index, si.receivers.r + 1)


def to_flat_dict(si: SolverInterface) -> dict:
    """Экспорт интерфейса в плоский dict с именами ключей по INTERFACE.md.

    Удобно для PythonAPI в C++ (pybind11): принимаем dict[str, ndarray|int|float].
    """
    # Время: конвертируем в (dt, TotalSteps)
    total_steps = si.time.time_steps

    out = {
        # Время
        "TimeBegin": si.time.time_begin,
        "TimeEnd": si.time.time_end ,
        "TimeSteps": si.time.time_steps,

        # Геометрия
        "DIM": int(si.geom.dim),
        "N": int(si.geom.n),
        "DOF": int(si.geom.dof),
        "NodesCoords": si.geom.nodes_coords,

        # Состояние
        "ST": int(si.state.st),
        "NodesState": si.state.nodes_state,
    }

    # Prime types
    out.update({
        "EPT": int(si.primes.ept),
        "ElementPrimeTypeNodes": si.primes.ept_nodes,
        "ElementPrimeTypeDims": si.primes.ept_dims,
        "EW": int(si.primes.epw),
        "ElementPrimeTypeWeight": si.primes.ept_weights,
        "ElementPrimeTypeWeightOffsets": si.primes.ept_weights_offsets,
        "ENS": int(si.primes.epns),
        "ElementPrimeTypeNablaShapes": si.primes.ept_nabla_shapes,
        "ElementPrimeTypeNablaShapesOffsets": si.primes.ept_nabla_shapes_offsets,
    })

    # Elements (composite)
    out.update({
        "E": int(si.elems.e),
        "ET": int(si.elems.et),
        "ElementsTypes": si.elems.elements_types,
        "ElementsTypesOffsets": si.elems.elements_types_offsets,
        "EN": int(si.elems.en),
        "ElementsNodes": si.elems.elements_nodes,
        "ElementsNodesOffsets": si.elems.elements_nodes_offsets,
    })

    # Materials
    out.update({
        "M": int(si.materials.m),
        "MaterialProps": si.materials.material_props,
    })

    # Node BCs
    out.update({
        "NC": int(si.node_bcs.nc),
        "NodesConditionTypes": si.node_bcs.nodes_bc_types,
        "NodesConditionBegin": si.node_bcs.nodes_bc_begin,
        "NodesConditionEnd": si.node_bcs.nodes_bc_end,
        "NodesConditionSteps": si.node_bcs.nodes_bc_steps,
        "NodesConditionNodes": si.node_bcs.nodes_bc_nodes,
        "NodesConditionTimeInterpType": si.node_bcs.nodes_bc_time_interp_type,
        "NCI": int(si.node_bcs.nci),
        "NodesConditionIndex": si.node_bcs.nodes_bc_index,
        "NodesConditionIndexOffsets": si.node_bcs.nodes_bc_index_offsets,
        "NCD": int(si.node_bcs.ncd),
        "NodesConditionData": si.node_bcs.nodes_bc_data,
        "NodesConditionDataOffsets": si.node_bcs.nodes_bc_data_offsets,
    })

    # Elem BCs
    out.update({
        "EC": int(si.elem_bcs.ec),
        "ElemsConditionTypes": si.elem_bcs.elems_condition_types,
        "ElemsConditionBegin": si.elem_bcs.elems_condition_begin,
        "ElemsConditionEnd": si.elem_bcs.elems_condition_end,
        "ElemsConditionSteps": si.elem_bcs.elems_condition_steps,
        "ElemsConditionNodes": si.elem_bcs.elems_condition_nodes,
        "ElemsConditionTimeInterpType": si.elem_bcs.elems_condition_time_interp_type,
        "ECI": int(si.elem_bcs.eci),
        "ElemsConditionIndex": si.elem_bcs.elems_condition_index,
        "ElemsConditionIndexOffsets": si.elem_bcs.elems_condition_index_offsets,
        "ECD": int(si.elem_bcs.ecd),
        "ElemsConditionData": si.elem_bcs.elems_condition_data,
        "ElemsConditionDataOffsets": si.elem_bcs.elems_condition_data_offsets,
    })

    # Receivers (output)
    out.update({
        "R": int(si.receivers.r),
        "ReceiverTypes": si.receivers.receivers_types,
        "ReceiverBegin": si.receivers.receivers_begin,
        "ReceiverEnd": si.receivers.receivers_end,
        "ReceiverSteps": si.receivers.receivers_steps,
        "RC": int(si.receivers.rc),
        "ReceiverComponents": si.receivers.receivers_components,
        "ReceiverComponentsOffsets": si.receivers.receivers_components_offsets,
        "RI": int(si.receivers.ri),
        "ReceiverIndex": si.receivers.receivers_index,
        "ReceiverIndexOffsets": si.receivers.receivers_index_offsets,
        "RD": int(si.receivers.rd),
        "ReceiverData": si.receivers.receivers_data,
        "ReceiverDataOffsets": si.receivers.receivers_data_offsets,
    })

    return out


