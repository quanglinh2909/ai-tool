import os

from piccolo.engine.sqlite import SQLiteEngine

# Lấy thư mục gốc sau khi loại bỏ 3 cấp
base_dir = os.path.abspath(os.path.dirname(__file__))  # Thư mục chứa file hiện tại
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(base_dir))) # Lùi lại 3 cấp

# Đường dẫn đến database
db_path = os.path.join(base_dir, "assets", "ai_tool.db")

# Tạo thư mục nếu chưa có
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Khởi tạo SQLiteEngine
DBSQLiteEngine = SQLiteEngine(path=db_path)
