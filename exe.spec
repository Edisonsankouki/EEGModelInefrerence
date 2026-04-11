# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
HOOKS_DIR = os.path.join(PROJECT_ROOT, "hooks")
HOOKSPATH = [HOOKS_DIR] if os.path.isdir(HOOKS_DIR) else []


def safe_collect_data_files(pkg, **kwargs):
    try:
        return collect_data_files(pkg, **kwargs)
    except Exception:
        return []


def safe_collect_dynamic_libs(pkg, **kwargs):
    try:
        return collect_dynamic_libs(pkg, **kwargs)
    except Exception:
        return []


def safe_collect_submodules(pkg, **kwargs):
    try:
        return collect_submodules(pkg, **kwargs)
    except Exception:
        return []


def safe_copy_metadata(pkg, recursive=False):
    try:
        return copy_metadata(pkg, recursive=recursive)
    except Exception:
        return []


def add_package_file(datas_list, pkg, relative_path, dest_dir=None):
    try:
        spec = importlib.util.find_spec(pkg)
    except Exception:
        spec = None

    if not spec or not spec.origin:
        return

    pkg_dir = os.path.dirname(spec.origin)
    src = os.path.join(pkg_dir, relative_path)
    if os.path.isfile(src):
        datas_list.append((src, dest_dir or pkg))


def filtered_submodules(pkg, exclude_prefixes=()):
    modules = safe_collect_submodules(pkg)
    if not exclude_prefixes:
        return modules
    return [
        name for name in modules
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in exclude_prefixes)
    ]


datas = []
for src_dir in ["model", "fine_tuning"]:
    abs_src = os.path.join(PROJECT_ROOT, src_dir)
    if os.path.isdir(abs_src):
        datas.append((abs_src, src_dir))

for pkg in [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "mne",
    "matplotlib",
    "pytorch_lightning",
    "lightning_fabric",
    "transformers",
    "timm",
    "rich",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "yaml",
    "tzdata",
]:
    datas += safe_collect_data_files(pkg)

# Some Lightning releases read version.info directly from the installed package
# at runtime, so keep these files explicitly even if generic collectors miss them.
add_package_file(datas, "pytorch_lightning", "version.info")
add_package_file(datas, "lightning_fabric", "version.info")

for pkg in [
    "setuptools",
    "packaging",
    "pytorch-lightning",
    "lightning-fabric",
    "lightning-utilities",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "numpy",
    "pandas",
    "mne",
    "matplotlib",
    "rich",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
]:
    datas += safe_copy_metadata(pkg, recursive=True)

binaries = []
for pkg in [
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "PyQt5",
]:
    binaries += safe_collect_dynamic_libs(pkg)

hiddenimports = [
    "pkg_resources",
    "setuptools",
    "torch.cuda",
    "numpy",
    "numpy.core._methods",
    "numpy.lib.format",
    "dateutil",
    "dateutil.tz",
    "tzdata",
    "yaml",
    "regex",
    "requests",
    "packaging",
    "filelock",
    "tqdm",
    "pytorch_lightning",
    "pytorch_lightning.strategies",
    "pytorch_lightning.strategies.ddp",
    "pytorch_lightning.strategies.single_device",
    "lightning_fabric",
    "lightning_utilities",
    "mne",
    "mne.io",
    "mne.io.edf",
    "mne.preprocessing",
    "mne.time_frequency",
    "mne.filter",
    "mne.channels",
    "mne.html_templates",
    "mne.html_templates._templates",
    "pandas",
    "torchvision",
    "torchvision.models",
    "timm",
    "timm.models",
    "timm.models.layers",
    "timm.models.registry",
    "transformers",
    "transformers.models",
    "transformers.models.gpt2",
    "transformers.models.gpt2.modeling_gpt2",
    "rich",
]
hiddenimports += filtered_submodules(
    "mne",
    exclude_prefixes=(
        "mne.commands",
        "mne.conftest",
        "mne.datasets",
        "mne.gui",
        "mne.simulation",
        "mne.tests",
        "mne.viz.backends._notebook",
    ),
)
hiddenimports = list(dict.fromkeys(hiddenimports))

excludes = [
    "torchaudio",
    "tensorflow",
    "jax",
    "jaxlib",
    "pytest",
    "IPython",
    "notebook",
]

a = Analysis(
    ["comcode.py"],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=HOOKSPATH,
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="gpt2ref",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="gpt2ref",
)
