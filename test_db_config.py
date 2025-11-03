"""
测试数据库配置的动态性
"""
from src.config import settings
import os

print("=" * 60)
print("📊 数据库配置测试")
print("=" * 60)

current_env = os.getenv('EMP_ENV', 'development')
print(f"\n当前环境: {current_env.upper()}")

print("\n【用户数据库配置】")
print(f"  数据库名: {settings.POSTGRES.USER_DB.NAME}")
print(f"  主机: {settings.POSTGRES.USER_DB.HOST}")
print(f"  端口: {settings.POSTGRES.USER_DB.PORT}")
print(f"  用户名: {settings.POSTGRES.USER_DB.USERNAME}")

print("\n【LangGraph 数据库配置】")
print(f"  数据库名: {settings.POSTGRES.LANGGRAPH_DB.NAME}")
print(f"  主机: {settings.POSTGRES.LANGGRAPH_DB.HOST}")
print(f"  URI: {settings.POSTGRES.LANGGRAPH_DB.URI}")

print("\n" + "=" * 60)
print("💡 测试环境变量覆盖:")
print("=" * 60)
print("""
# 覆盖用户数据库名
$env:EMP_CONF_POSTGRES__USER_DB__NAME="test_db"
python test_db_config.py

# 覆盖 LangGraph 数据库名
$env:EMP_CONF_POSTGRES__LANGGRAPH_DB__NAME="test_langgraph"
python test_db_config.py

# 切换到生产环境
$env:EMP_ENV="production"
python test_db_config.py
""")

