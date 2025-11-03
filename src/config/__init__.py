from pathlib import Path
from dynaconf import Dynaconf
import os

_BASE_DIR = Path(__file__).parent.parent.parent  # 得到项目目录

# 根据环境变量选择配置文件（最简单的实现）
env = os.getenv('EMP_ENV', 'development')  # 默认 development
config_file = f'{env}.yml'  # development.yml 或 production.yml

settings = Dynaconf(
    envvar_prefix="EMP_CONF",  # 环境变量前缀
    settings_files=[Path(__file__).parent / config_file],  # 配置文件路径
    lowercase_read=False,  # 不将配置文件中的键名转换为小写
    base_dir=_BASE_DIR,  # 项目目录
)
