"""
Import all of the Tables subclasses in your app here, and register them with
the APP_CONFIG.
"""

import os
import inspect
import importlib
import pkgutil
from typing import List

from piccolo.conf.apps import AppConfig
from piccolo.table import Table

# Thư mục hiện tại
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Hàm để lấy tất cả các Table subclasses từ app.models
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

# Lấy tất cả các Table classes
table_classes = get_all_table_classes()

# Hiển thị các classes đã tìm thấy (cho debugging)
# print(f"Đã tìm thấy {len(table_classes)} Table classes: {[cls.__name__ for cls in table_classes]}")

# Tạo AppConfig với các table classes đã tìm được
APP_CONFIG = AppConfig(
    app_name="sqlite",
    migrations_folder_path=os.path.join(
        CURRENT_DIRECTORY, "piccolo_migrations"
    ),
    table_classes=table_classes,
    migration_dependencies=[],
    commands=[],
)