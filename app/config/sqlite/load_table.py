from typing import List
import inspect
import importlib
import pkgutil

from fastapi import FastAPI
from piccolo.table import Table

def get_all_table_classes() -> List[Table]:
    import app.models

    table_classes = []

    # Duyệt qua tất cả các module trong package app.models
    for _, module_name, _ in pkgutil.iter_modules(app.models.__path__, app.models.__name__ + "."):
        try:
            # Import module
            module = importlib.import_module(module_name)

            # Tìm tất cả các class trong module là subclass của Table
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Table) and obj != Table:
                    table_classes.append(obj)
        except ImportError as e:
            print(f"Không thể import module {module_name}: {e}")

    return table_classes

async def get_all_tables() -> List[Table]:
    """
    Lấy tất cả các bảng trong ứng dụng FastAPI.
    """
    table_classes = get_all_table_classes()
    tables = []

    for table_class in table_classes:
        if table_class.__name__ != "BaseModel":
            print("Creating table:", table_class.__name__)
            await table_class.create_table(if_not_exists=True)
            tables.append(table_class)

    return tables