"""SEMCMPU: простая CPU-обёртка прототипа SEMGPU на numpy.

Содержит:
- описание интерфейса входных/выходных массивов (см. interface.SolverInterface)
- однопоточную оболочку решателя с заглушками трёх ядер (см. solver.SEMCPUSolver)

Назначение: отладка интерфейса и подготовка тестов для GPU-версий.
"""

from .interface import (
    SolverTime,
    NodesGeometryInput,
    NodesStateInput,
    PrimeElementTypesInput,
    CommonElementsInput,
    MaterialsInput,
    NodeBCsInput,
    ElemBCsInput,
    ReceiversOutput,
    SolverInterface,
    validate_interface,
)

from .solver import SEMCPUSolver
from .gpu_solver import SEMGPUSolver

__all__ = [
    "SolverTime",
    "NodesGeometryInput",
    "NodesStateInput",
    "PrimeElementTypesInput",
    "CommonElementsInput",
    "MaterialsInput",
    "NodeBCsInput",
    "ElemBCsInput",
    "ReceiversOutput",
    "SolverInterface",
    "validate_interface",
    "SEMCPUSolver",
    "SEMGPUSolver",
]



