"""
查看 user_db 数据库中的用户数据
"""
from src.db import sm
from src.db.system_mgt.models import UserModel

print("=" * 80)
print("📊 user_db 数据库 - t_usermodel 表数据")
print("=" * 80)

# 创建数据库会话
session = sm()

try:
    # 查询所有用户
    users = session.query(UserModel).all()
    
    if not users:
        print("\n❌ 表中没有数据（还没有注册用户）\n")
    else:
        print(f"\n✅ 共找到 {len(users)} 个用户：\n")
        
        for user in users:
            print(f"ID: {user.id}")
            print(f"用户名: {user.username}")
            print(f"密码（加密）: {user.password[:20]}...")  # 只显示前20个字符
            print(f"手机号: {user.phone or '未设置'}")
            print(f"邮箱: {user.email or '未设置'}")
            print(f"真实姓名: {user.real_name or '未设置'}")
            print(f"头像: {user.icon}")
            print(f"部门ID: {user.dept_id or '未设置'}")
            print(f"创建时间: {user.create_time}")
            print(f"更新时间: {user.update_time}")
            print("-" * 80)
    
    print("\n" + "=" * 80)
    
finally:
    session.close()

