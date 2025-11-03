"""
数据库初始化脚本
创建所有的数据库表
"""
from src.db import DBModelBase, engine
from src.db.system_mgt.models import UserModel

print("🔧 开始创建数据库表...")

# 创建所有表
DBModelBase.metadata.create_all(bind=engine)

print("✅ 数据库表创建成功！")
print(f"📋 已创建的表：{list(DBModelBase.metadata.tables.keys())}")

