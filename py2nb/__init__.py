#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер между Python файлами с ячейками и Jupyter ноутбуками

Логика конвертации:
- py -> ipynb: многострочные комментарии \"\"\" становятся markdown ячейками,
  функции и классы - отдельными code ячейками, остальной код группируется
- ipynb -> py: аналогично в обратном направлении, outputs игнорируются
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


# Экспортируем функцию main для использования в __main__.py
def main():
    """Главный интерфейс конвертера"""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python -m py2nb input.py [output.ipynb]    # py -> ipynb")
        print("  python -m py2nb input.ipynb [output.py]    # ipynb -> py")
        print("Если выходной файл не указан, он генерируется автоматически")
        sys.exit(1)

    input_file = sys.argv[1]

    # Если указан второй аргумент, используем его
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Автоматическая генерация выходного файла
        input_path = Path(input_file)
        if input_path.suffix == '.py':
            output_file = str(input_path.with_suffix('.ipynb'))
        elif input_path.suffix == '.ipynb':
            output_file = str(input_path.with_suffix('.py'))
        else:
            print(f"Неподдерживаемый формат файла: {input_path.suffix}")
            print("Поддерживаются только .py и .ipynb файлы")
            sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Файл {input_file} не найден")
        sys.exit(1)

    try:
        if input_path.suffix == '.py':
            # Конвертация Python -> Jupyter
            converter = PythonToIPythonConverter()
            cells = converter.parse_python_file(str(input_path))
            converter.generate_ipynb(str(output_path))
            print(f"Конвертировано {len(cells)} ячеек: {input_file} → {output_file}")

        elif input_path.suffix == '.ipynb':
            # Конвертация Jupyter -> Python
            converter = IPythonToPythonConverter()
            python_code = converter.parse_ipynb_file(str(input_path))
            converter.save_python_file(python_code, str(output_path))
            print(f"Конвертировано: {input_file} → {output_file}")

        else:
            print("Неподдерживаемый формат файла")
            sys.exit(1)

    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
        sys.exit(1)


class PythonToIPythonConverter:
    """Конвертер Python файлов в Jupyter ноутбуки"""

    def __init__(self):
        self.cells = []

    def parse_python_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Парсит Python файл и выделяет блоки для конвертации в ячейки"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Используем AST для парсинга структур
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Ошибка синтаксиса в файле {filepath}: {e}")
            return []

        self.cells = []
        self._extract_cells_from_ast(tree, content)

        return self.cells

    def _extract_cells_from_ast(self, tree: ast.AST, content: str):
        """Извлекает ячейки из AST дерева"""
        lines = content.split('\n')

        # Находим все функции, классы и многострочные строки
        functions = []
        classes = []
        multiline_strings = []

        # Собираем все функции для определения вложенности
        all_functions = []
        all_classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                all_functions.append((node.lineno, node.end_lineno, node.name, node))
            elif isinstance(node, ast.ClassDef):
                all_classes.append((node.lineno, node.end_lineno, node.name, node))

        # Фильтруем только top-level функции (не вложенные)
        for func_start, func_end, func_name, func_node in all_functions:
            # Проверяем, является ли функция вложенной
            is_nested = False
            for parent_start, parent_end, _, _ in all_functions + all_classes:
                if parent_start < func_start < parent_end and (func_start, func_end, func_name, func_node) != (parent_start, parent_end, _, _):
                    is_nested = True
                    break
            if not is_nested:
                functions.append((func_start, func_end, func_name))

        # Фильтруем только top-level классы
        for class_start, class_end, class_name, class_node in all_classes:
            # Проверяем, является ли класс вложенным
            is_nested = False
            for parent_start, parent_end, _, _ in all_functions + all_classes:
                if parent_start < class_start < parent_end and (class_start, class_end, class_name, class_node) != (parent_start, parent_end, _, _):
                    is_nested = True
                    break
            if not is_nested:
                classes.append((class_start, class_end, class_name))

        # Находим многострочные строки (только top-level, не внутри функций/классов)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                if node.lineno != node.end_lineno:  # Многострочный
                    # Проверяем, является ли это docstring (находится внутри функции/класса)
                    is_docstring = False
                    parent = getattr(node, '_parent', None)
                    if parent and isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                        is_docstring = True

                    # Также проверяем по положению в коде
                    for func_start, func_end, _, _ in all_functions:
                        if func_start < node.lineno < func_end:
                            is_docstring = True
                            break
                    for class_start, class_end, _, _ in all_classes:
                        if class_start < node.lineno < class_end:
                            is_docstring = True
                            break

                    if not is_docstring:
                        multiline_strings.append((node.lineno, node.end_lineno))

        # Сортируем все блоки по начальной строке
        all_blocks = []
        for start, end, name in functions:
            all_blocks.append(('function', start, end, name))
        for start, end, name in classes:
            all_blocks.append(('class', start, end, name))
        for start, end in multiline_strings:
            all_blocks.append(('multiline_string', start, end, None))

        all_blocks.sort(key=lambda x: x[1])  # Сортировка по начальной строке

        # Создаем ячейки
        current_pos = 0
        for block_type, start, end, name in all_blocks:
            # Добавляем код перед блоком
            if start > current_pos + 1:
                code_lines = lines[current_pos:start-1]
                if code_lines and any(line.strip() for line in code_lines):
                    self.cells.append({
                        'type': 'code',
                        'name': 'code',
                        'content': '\n'.join(code_lines)
                    })

            # Добавляем сам блок
            if block_type in ('function', 'class'):
                cell_lines = lines[start-1:end]
                self.cells.append({
                    'type': 'code',
                    'name': f"{block_type} {name}",
                    'content': '\n'.join(cell_lines)
                })
                current_pos = end  # Обновляем позицию сразу после добавления функции/класса
                continue  # Пропускаем обновление current_pos в конце цикла
            elif block_type == 'multiline_string':
                # Проверяем, является ли это markdown ячейкой (начинается с r""")
                first_line = lines[start-1] if start-1 < len(lines) else ""
                is_markdown_cell = first_line.strip().startswith('r\"\"\"')

                if is_markdown_cell:
                    # Это markdown ячейка - создаем отдельную ячейку
                    docstring_lines = lines[start-1:end]
                    # Убираем r"""
                    if docstring_lines and docstring_lines[0].strip().startswith('r\"\"\"'):
                        docstring_lines[0] = docstring_lines[0].replace('r\"\"\"', '', 1)
                    if docstring_lines and docstring_lines[-1].strip().endswith('\"\"\"'):
                        docstring_lines[-1] = docstring_lines[-1].rsplit('\"\"\"', 1)[0]

                    # Очищаем пустые строки
                    while docstring_lines and not docstring_lines[0].strip():
                        docstring_lines.pop(0)
                    while docstring_lines and not docstring_lines[-1].strip():
                        docstring_lines.pop()

                    if docstring_lines:
                        self.cells.append({
                            'type': 'markdown',
                            'name': 'markdown',
                            'content': '\n'.join(docstring_lines)
                        })
                    current_pos = end  # Обновляем позицию после markdown ячейки
                    continue  # Пропускаем обновление current_pos в конце цикла
                # Если это обычный """ или docstring, игнорируем его -
                # он будет включен в ячейку функции/класса или обычного кода

            current_pos = end

        # Добавляем оставшийся код
        if current_pos < len(lines):
            remaining_lines = lines[current_pos:]
            if remaining_lines and any(line.strip() for line in remaining_lines):
                self.cells.append({
                    'type': 'code',
                    'name': 'code',
                    'content': '\n'.join(remaining_lines)
                })

        # Если ничего не найдено, добавляем весь файл как одну ячейку
        if not self.cells:
            self.cells.append({
                'type': 'code',
                'name': 'main',
                'content': content
            })

    def _find_block_end(self, lines: List[str], start_line_idx: int) -> int:
        """Находит конец блока (функции/класса) по отступам"""
        if start_line_idx >= len(lines):
            return len(lines)

        # Определяем базовый отступ блока
        base_indent = len(lines[start_line_idx]) - len(lines[start_line_idx].lstrip())

        # Ищем конец блока - строку с тем же или меньшим отступом
        for i in range(start_line_idx + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Пропускаем пустые строки
                current_indent = len(line) - len(line.lstrip())
                # Если отступ меньше или равен базовому - это конец блока
                if current_indent <= base_indent:
                    return i

        return len(lines)  # Если не нашли конец, возвращаем конец файла

    def _add_code_cell(self, lines: List[str]):
        """Добавляет code ячейку из списка строк"""
        # Очищаем пустые строки в начале и конце
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        if lines:
            self.cells.append({
                'type': 'code',
                'name': 'code',
                'content': '\n'.join(lines)
            })

    def generate_ipynb(self, output_path: str):
        """Генерирует .ipynb файл из ячеек"""
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        for cell_data in self.cells:
            if cell_data['type'] == 'markdown':
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": cell_data['content'].split('\n')
                }
            else:  # code
                cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": cell_data['content'].split('\n')
                }
            notebook["cells"].append(cell)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)


class IPythonToPythonConverter:
    """Конвертер Jupyter ноутбуков в Python файлы"""

    def __init__(self):
        self.python_code = []

    def parse_ipynb_file(self, filepath: str) -> str:
        """Парсит .ipynb файл и конвертирует в Python код"""
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        self.python_code = []

        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', 'code')
            source_lines = cell.get('source', [])

            # Преобразуем source из списка в строку
            if isinstance(source_lines, list):
                # Обрабатываем каждую строку, убирая лишние \n в конце
                processed_lines = []
                for line in source_lines:
                    # Убираем \n в конце каждой строки, если он есть
                    if line.endswith('\n'):
                        line = line[:-1]
                    processed_lines.append(line)

                # Убираем пустые строки в начале
                while processed_lines and not processed_lines[0].strip():
                    processed_lines.pop(0)

                # Убираем пустые строки в конце
                while processed_lines and not processed_lines[-1].strip():
                    processed_lines.pop()

                content = '\n'.join(processed_lines)
            else:
                content = source_lines

            if cell_type == 'markdown':
                # Оборачиваем markdown в многострочный комментарий
                if content.strip():
                    self.python_code.append(f'\"\"\"\n{content}\n\"\"\"')

            elif cell_type == 'code':
                # Добавляем код как есть
                if content.strip():
                    self.python_code.append(content)

            # Добавляем пустую строку для разделения между ячейками (кроме последней)
            if cell != notebook.get('cells', [])[-1]:
                self.python_code.append('')

        return '\n'.join(self.python_code)

    def save_python_file(self, content: str, output_path: str):
        """Сохраняет Python код в файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


if __name__ == "__main__":
    main()
