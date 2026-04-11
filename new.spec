# -*- mode: python ; coding: utf-8 -*-
# gpt2ref_cpu.spec
# CPU 版（兼容 torch.cuda 命名空间导入）、torchvision、timm、pytorch_lightning、transformers、mne、pandas
# 重点修复：pkg_resources / setuptools metadata / lightning 依赖链

import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# -----------------------------
# 基础路径
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()
HOOKS_DIR = os.path.join(PROJECT_ROOT, 'hooks')
HOOKSPATH = [HOOKS_DIR] if os.path.isdir(HOOKS_DIR) else []

# -----------------------------
# 安全收集工具（避免某些包不存在时报错）
# -----------------------------
def safe_collect_data_files(pkg, **kwargs):
    try:
        return collect_data_files(pkg, **kwargs)
    except Exception as e:
        print(f"[spec] skip data files for {pkg}: {e}")
        return []

def safe_collect_dynamic_libs(pkg, **kwargs):
    try:
        return collect_dynamic_libs(pkg, **kwargs)
    except Exception as e:
        print(f"[spec] skip dynamic libs for {pkg}: {e}")
        return []

def safe_collect_submodules(pkg, **kwargs):
    try:
        return collect_submodules(pkg, **kwargs)
    except Exception as e:
        print(f"[spec] skip submodules for {pkg}: {e}")
        return []

def safe_copy_metadata(pkg, recursive=False):
    try:
        return copy_metadata(pkg, recursive=recursive)
    except Exception as e:
        print(f"[spec] skip metadata for {pkg}: {e}")
        return []

# -----------------------------
# 数据文件（datas）
# -----------------------------
datas = []

# 你本地资源目录（存在才打包）
for src_dir in ['model', 'fine_tuning']:
    abs_src = os.path.join(PROJECT_ROOT, src_dir)
    if os.path.isdir(abs_src):
        datas.append((abs_src, src_dir))
    else:
        print(f"[spec] local resource folder not found, skip: {abs_src}")

# 常见包数据文件
for pkg in [
    'torch',
    'torchvision',
    'numpy',
    'pandas',
    'mne',
    'matplotlib',
    'transformers',
    'timm',
    'rich',
    'tokenizers',
    'safetensors',
    'huggingface_hub',
    'yaml',        # PyYAML
    'tzdata',      # 如果没装会自动跳过
]:
    datas += safe_collect_data_files(pkg)

# metadata（很关键：pkg_resources / entry points / version checks）
# setuptools <81 时会提供 pkg_resources；这里显式复制 metadata
for pkg in [
    'setuptools',
    'pip',  # 可选；有时帮助 pkg_resources 环境检查
    'packaging',
    'pytorch-lightning',
    'lightning-fabric',
    'lightning-utilities',
    'torch',
    'torchvision',
    'transformers',
    'timm',
    'numpy',
    'pandas',
    'mne',
    'matplotlib',
    'rich',
    'tokenizers',
    'safetensors',
    'huggingface-hub',
]:
    datas += safe_copy_metadata(pkg, recursive=True)

# -----------------------------
# 动态库（binaries）
# -----------------------------
binaries = []

# 核心动态库
for pkg in [
    'torch',
    'torchvision',
    'numpy',
    'pandas',
    'matplotlib',
    'PyQt5',
    'scipy',      # 如果未安装会跳过
]:
    binaries += safe_collect_dynamic_libs(pkg)

# timm 通常是纯 Python，这里不强制；存在动态库则收集，否则跳过
binaries += safe_collect_dynamic_libs('timm')

# -----------------------------
# 隐式导入（hiddenimports）
# -----------------------------
hiddenimports = []

# 你的依赖链核心
for pkg in [
    'pytorch_lightning',
    'lightning_fabric',
    'lightning_utilities',
    'transformers',
    'mne',
    'rich',
    'torchvision',
    'timm',
    'pandas',
]:
    hiddenimports += safe_collect_submodules(pkg)

# 显式兜底（你之前报错相关 + 常见运行时链路）
hiddenimports += [
    # pkg_resources / setuptools 兼容
    'pkg_resources',
    'setuptools',

    # torch / numpy 常见隐式模块
    'torch.cuda',              # 即使 CPU 版，也有代码会 import torch.cuda 命名空间
    'numpy',
    'numpy.core._methods',
    'numpy.lib.format',

    # pandas / 时区
    'dateutil',
    'dateutil.tz',
    'tzdata',

    # transformers 常见依赖（按需可删）
    'yaml',
    'regex',
    'requests',
    'packaging',
    'filelock',
    'tqdm',
]

# 去重（保持顺序）
hiddenimports = list(dict.fromkeys(hiddenimports))

# -----------------------------
# 排除项（按需调整，避免拉入过多无关依赖）
# -----------------------------
excludes = [
    'torchaudio',   # 你明确不用
    'tensorflow',
    'jax',
    'jaxlib',
    'pytest',
    'IPython',
    'notebook',
]

# -----------------------------
# Analysis
# -----------------------------
a = Analysis(
    ['comcode.py'],           # 如果入口不是这个，请改
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=HOOKSPATH,      # 没有 hooks 目录时会自动 []
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

# -----------------------------
# PYZ / EXE / COLLECT
# -----------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gpt2ref',
    debug=False,                    # 排障时可改 True
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                      # 建议保持 False
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='gpt2ref'
)