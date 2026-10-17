"""在受限 Windows 环境中逐单元执行项目 Notebook，并写回真实输出。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import uuid

import nbformat
from IPython.utils.capture import capture_output


def execute_notebook(path: Path, working_directory: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    namespace: dict[str, object] = {
        "__name__": "__main__",
        "__file__": str(path),
    }
    original_directory = Path.cwd()
    original_sys_path = list(sys.path)
    try:
        import os

        os.chdir(working_directory)
        sys.path.insert(0, str(working_directory))
        execution_count = 0
        for cell_index, cell in enumerate(notebook.cells):
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]
            if cell.cell_type != "code":
                continue
            execution_count += 1
            cell.execution_count = execution_count
            cell.outputs = []
            try:
                with capture_output() as captured:
                    exec(compile(cell.source, f"{path.name}:cell-{cell_index}", "exec"), namespace)
            except Exception as exc:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "error",
                        ename=type(exc).__name__,
                        evalue=str(exc),
                        traceback=traceback.format_exc().splitlines(),
                    )
                )
                nbformat.write(notebook, path)
                raise RuntimeError(f"Notebook {path.name} 第 {cell_index} 个单元执行失败。") from exc
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data",
                        data=output.data,
                        metadata=output.metadata or {},
                    )
                )
        notebook.metadata.setdefault("execution", {})["engine"] = "project_inprocess_runner"
        notebook.metadata["execution"]["working_directory"] = str(working_directory)
        nbformat.write(notebook, path)
    finally:
        os.chdir(original_directory)
        sys.path[:] = original_sys_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    args = parser.parse_args()
    for notebook_path in args.notebooks:
        execute_notebook(notebook_path.resolve(), args.working_directory.resolve())
        print(f"已执行：{notebook_path}")


if __name__ == "__main__":
    main()
