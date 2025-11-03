# 定义数据库连接的URL
from datetime import datetime

from sqlalchemy import URL, create_engine, DateTime, func
from sqlalchemy.orm import sessionmaker, scoped_session, DeclarativeBase, declared_attr, Mapped, mapped_column

from src.config import settings

# 使用用户管理数据库配置
db_config = settings.POSTGRES.USER_DB if hasattr(settings, 'POSTGRES') else settings.MYSQL

# 创建数据库连接 URL
url = URL(
    drivername=db_config.DRIVER,
    username=db_config.get('USERNAME'),
    password=db_config.get('PASSWORD'),
    host=db_config.get('HOST'),
    port=db_config.get('PORT'),
    database=db_config.get('NAME'),
    query=db_config.get('QUERY', {}),  # PostgreSQL 使用空字典，MySQL 使用 {charset: utf8mb4}
)

# 使用 URL 创建数据库引擎
engine = create_engine(url, echo=True, future=True, pool_size=10)

sm = sessionmaker(bind=engine, autoflush=True, autocommit=False)


class DBModelBase(DeclarativeBase):
    """ 定义一系列的可以映射的公共属性，此类是所有数据库模型类的父类"""

    @declared_attr.directive
    def __tablename__(cls) -> str:
        return 't_'+cls.__name__.lower()  # 未来所有的模型中：表名就是：t_当前模型类名字

    # PostgreSQL 不需要指定引擎（MySQL 才需要 mysql_engine）
    # __table_args__ = {"mysql_engine": "InnoDB"}  # 仅 MySQL 使用
    
    #  如果为true，则ORM将在插入或更新之后立即获取服务器生成的默认值的值
    __mapper_args__ = {"eager_defaults": True}

    # 所有的模型类，都有的属性和字段映射
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now(), comment='记录的创建时间')
    update_time: Mapped[datetime] = mapped_column(DateTime, insert_default=func.now(), onupdate=func.now(),
                                                  comment='记录的最后一次修改时间')