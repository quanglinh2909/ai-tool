from piccolo.conf.apps import AppRegistry

from app.config.sqlite.DBSQLiteEngine import DBSQLiteEngine

DB = DBSQLiteEngine

# Đăng ký ứng dụng Piccolo
APP_REGISTRY = AppRegistry(apps=['app.config.sqlite.piccolo_app'])
