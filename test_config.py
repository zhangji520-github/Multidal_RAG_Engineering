"""
配置读取测试脚本
演示如何在任何文件中读取 settings
"""

# ✅ 第 1 步：导入 settings
from src.config import settings

print("=" * 60)
print("📖 配置读取演示")
print("=" * 60)

# ✅ 第 2 步：读取简单配置
print("\n【简单配置】")
print(f"HOST: {settings.HOST}")
print(f"PORT: {settings.PORT}")
print(f"LOG_LEVEL: {settings.LOG_LEVEL}")

# ✅ 第 3 步：读取列表配置
print("\n【列表配置】")
print(f"ORIGINS: {settings.ORIGINS}")
print(f"WHITE_LIST: {settings.WHITE_LIST}")

# ✅ 第 4 步：读取嵌套配置（PostgreSQL）
print("\n【嵌套配置 - PostgreSQL】")
print(f"POSTGRES.HOST: {settings.POSTGRES.USER_DB.HOST}")
print(f"POSTGRES.PORT: {settings.POSTGRES.USER_DB.PORT}")
print(f"POSTGRES.USERNAME: {settings.POSTGRES.USER_DB.USERNAME}")
print(f"POSTGRES.PASSWORD: {settings.POSTGRES.USER_DB.PASSWORD}")
print(f"POSTGRES.URI: {settings.POSTGRES.USER_DB.URI}")

# ✅ 第 5 步：读取深层嵌套配置（Milvus）
print("\n【深层嵌套配置 - Milvus】")
print(f"MILVUS.URI: {settings.MILVUS.URI}")
print(f"MILVUS.USERNAME: {settings.MILVUS.USERNAME}")
print(f"MILVUS.COLLECTIONS.KNOWLEDGE: {settings.MILVUS.COLLECTIONS.KNOWLEDGE}")
print(f"MILVUS.COLLECTIONS.CONTEXT: {settings.MILVUS.COLLECTIONS.CONTEXT}")

# ✅ 第 6 步：读取 JWT 配置
print("\n【JWT 配置】")
print(f"JWT_SECRET_KEY: {settings.JWT_SECRET_KEY[:20]}...")  # 只显示前20个字符
print(f"ALGORITHM: {settings.ALGORITHM}")
print(f"ACCESS_TOKEN_EXPIRE_MINUTES: {settings.ACCESS_TOKEN_EXPIRE_MINUTES}")

# ✅ 第 7 步：测试类型自动转换
print("\n【类型自动转换】")
print(f"PORT 的类型: {type(settings.PORT)} = {settings.PORT}")
print(f"ACCESS_TOKEN_EXPIRE_MINUTES 的类型: {type(settings.ACCESS_TOKEN_EXPIRE_MINUTES)} = {settings.ACCESS_TOKEN_EXPIRE_MINUTES}")
print(f"ORIGINS 的类型: {type(settings.ORIGINS)}")

print("\n" + "=" * 60)
print("✅ 配置读取成功！")
print("=" * 60)

# ✅ 第 8 步：演示如何在函数中使用
def get_database_connection():
    """模拟数据库连接"""
    db_config = {
        'host': settings.POSTGRES.HOST,
        'port': settings.POSTGRES.PORT,
        'user': settings.POSTGRES.USERNAME,
        'password': settings.POSTGRES.PASSWORD,
        'database': settings.POSTGRES.NAME
    }
    print("\n【函数中使用配置】")
    print(f"数据库连接参数: {db_config}")
    return db_config

# 调用函数
get_database_connection()

