# gpt2ref

这个仓库是一个基于 EDF 文件的脑电推理项目。

主用途是：

- 读取 `.edf` 脑电文件
- 按不同任务做通道匹配、重采样、切片和特征准备
- 调用对应模型或规则分析逻辑完成推理
- 输出面向终端的结果文本
- 支持打包为 Windows 下可直接运行的 `exe`

当前以这个内层 `gpt2ref/` 目录作为唯一主仓库。

## 目录结构

建议重点关注这些文件和目录：

- `comcode.py`
  统一命令行入口，负责按任务名分发到各模块
- `seizure.py`
  癫痫检测
- `sleep.py`
  睡眠分期
- `spikeNet.py`
  癫痫波检测
- `report.py`
  EEG 报告生成
- `ICA.py`
  ICA 伪迹分解与频段占比分析
- `Cbra.py`
  多标签筛查，当前主要输出抑郁 / 焦虑 / 癫痫风险
- `RSN_seizure.py`
  基于 RSN 的癫痫检测
- `model/`
  模型定义与 GPT2/LaBraM/RSN 等结构代码
- `fine_tuning/`
  模型权重
- `hooks/`
  PyInstaller 打包辅助 hook
- `exe.spec`
  当前实际使用的 PyInstaller 打包配置
- `build_exe_venv.bat`
  在项目内 `.venv` 中打包 EXE 的批处理脚本

## 运行入口

统一入口：

```bash
python comcode.py <model> <edf_path>
```

例如：

```bash
python comcode.py seizure 20370.edf
python comcode.py sleep SC4102E0-PSG.edf
python comcode.py detspike aaaaafoe_00000001.edf
python comcode.py report 20370.edf
python comcode.py ica 20370.edf
python comcode.py cba 20370.edf
python comcode.py rsn 20370.edf
```

支持的 `model` 参数：

- `seizure`
- `sleep`
- `detspike`
- `report` 或 `Report_generation`
- `ica`
- `cba`
- `rsn`

## 项目推理逻辑

### 1. 总体流程

大多数模块都遵循这条主链路：

1. 读取 EDF
2. 匹配目标通道
3. 重采样或切片
4. 做归一化或频谱相关预处理
5. 加载 `fine_tuning/` 下的模型权重
6. 执行推理
7. 输出文本结果

不同任务的差异主要在：

- 需要的通道不同
- 是否切片
- 使用深度模型还是规则分析
- 最终输出格式不同

### 2. 入口分发逻辑

[comcode.py](/mnt/c/Users/jiang/Downloads/gpt2ref/gpt2ref/comcode.py:1) 只做一件事：按任务名导入对应模块并执行。

当前映射关系：

- `detspike` -> `spikeNet.detspike`
- `seizure` -> `seizure.supervised`
- `sleep` -> `sleep.sleep`
- `report` / `Report_generation` -> `report.Report_generation`
- `ica` -> `ICA.ICAArtifactRemover().fit_transform`
- `cba` -> `Cbra.Cbramod_run`
- `rsn` -> `RSN_seizure.supervised_rsn`

### 3. 各模块功能

#### `seizure.py`

用途：

- 做癫痫检测二分类

主要逻辑：

- 从 EDF 中构造 16 个双极导联
- 支持常见通道别名，例如 `T3/T7`、`T4/T8`、`T5/P7`、`T6/P8`
- 重采样到 `200 Hz`
- 按通道做标准化
- 加载 `fine_tuning/seizure.ckpt`
- 通过 `gpt4ts` 模型推理
- 输出检测结果

特点：

- 通道校验做得比较严格
- 通道缺失时会输出结构化错误信息

#### `sleep.py`

用途：

- 做睡眠分期

主要逻辑：

- 从 EDF 中寻找 `Fpz / Cz / Pz / Oz` 相关通道
- 至少要求命中 2 个目标通道
- 以 `30 秒` 为单位切片
- 加载 `fine_tuning/gpt4ts-sleep.ckpt`
- 对每段做分期推理
- 对结果做平滑
- 输出整段睡眠时长和各分期持续时间

特点：

- 对整晚 PSG 文件比较慢
- 在 CPU 上常常需要几分钟

#### `spikeNet.py`

用途：

- 做癫痫波检测

主要逻辑：

- 从 EDF 中提取指定通道
- 按 `5 秒` 切片
- 对每片段重采样到固定长度
- 加载 `fine_tuning/spike.ckpt`
- 推理后把结果写到 `classifications_spike.csv`
- 再把连续片段合并成持续时间描述
- 输出每种类别的时长和 normal 占比

分类标签包括：

- `normal`
- `SPSW`
- `GPED`
- `PLED`

#### `report.py`

用途：

- 生成 EEG 报告文本

主要逻辑：

- 读取 EDF 并挑选合适分析窗口
- 做频谱分析、功率分析、对称性分析、调节性分析
- 生成主导节律、慢波、快波、左右对称性等结论
- 进一步输出结构化风险评分和说明文字
- 最终输出一段较长的文本报告

特点：

- 这个模块不是单纯模型分类
- 更接近“规则分析 + 指标映射 + 报告生成”

#### `ICA.py`

用途：

- 对 EDF 做 ICA 分解
- 统计每个独立成分在不同频段的功率占比

主要逻辑：

- 读取 EDF
- 滤波
- 可选重采样
- 用 `mne.preprocessing.ICA` 拟合
- 输出每个 IC 的脑电 / 慢波 / 心电 / 眼动 / 肌电 / 工频 / 噪声占比

注意：

- `fastica` 依赖 `scikit-learn`
- 当前 `.venv` 如果没有安装 `scikit-learn`，这个模块会失败

#### `Cbra.py`

用途：

- 做多标签脑电筛查

当前输出重点：

- 抑郁风险
- 焦虑风险
- 癫痫风险

主要逻辑：

- 读取 EDF
- 强约束匹配 16 个通道
- 统一重采样到 `200 Hz`
- 切成固定 patch 结构
- 加载 `fine_tuning/cbra1.ckpt`
- 通过 `CBraMod` 推理
- 根据阈值输出中文筛查报告

#### `RSN_seizure.py`

用途：

- 基于 RSN 模型做癫痫检测

主要逻辑：

- 同样先构造 16 个双极导联
- 重采样并 reshape 成 `16 x 10 x 200`
- 加载 `fine_tuning/RSN.pth`
- 输出检测结果

### 4. 模型与资源

仓库里的主要运行资源：

- `fine_tuning/seizure.ckpt`
- `fine_tuning/gpt4ts-sleep.ckpt`
- `fine_tuning/spike.ckpt`
- `fine_tuning/RSN.pth`
- `fine_tuning/cbra1.ckpt`
- `model/gpt2/` 下的 GPT2 tokenizer / config 等资源

这些资源在源码运行和 EXE 打包时都需要保留。

## 已验证的运行情况

在项目自带 `.venv` 中，已实际验证：

- `seizure 20370.edf` 可运行
- `rsn 20370.edf` 可运行
- `cba 20370.edf` 可运行
- `report 20370.edf` 可运行
- `sleep SC4102E0-PSG.edf` 可运行，但耗时较长
- `detspike aaaaafoe_00000001.edf` 可运行，但耗时较长

已知问题：

- `ica` 依赖 `scikit-learn`，未安装时会报错
- Windows 控制台下中文输出可能乱码
- `pytorch_lightning` / `timm` 运行时会有一些警告

## 打包逻辑

### 1. 打包入口

推荐使用：

```bat
build_exe_venv.bat
```

这个脚本会：

1. 进入项目目录
2. 检查 `.venv\Scripts\python.exe` 是否存在
3. 清理旧的 `build\exe` 和 `dist\gpt2ref`
4. 使用 `.venv` 中的 Python 运行 PyInstaller

实际执行命令等价于：

```bat
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean exe.spec
```

### 2. 打包配置

当前实际使用的是：

- [exe.spec](/mnt/c/Users/jiang/Downloads/gpt2ref/gpt2ref/exe.spec:1)

`new.spec` 目前更像是调试/备选版本，不是默认入口。

### 3. `exe.spec` 的作用

`exe.spec` 做了这些事：

- 入口脚本设为 `comcode.py`
- 把 `model/` 和 `fine_tuning/` 整个加入打包数据
- 收集 `torch`、`torchvision`、`numpy`、`pandas`、`mne`、`matplotlib`、`pytorch_lightning`、`transformers`、`timm` 等依赖的数据文件
- 收集必要动态库
- 显式添加常见 hidden imports
- 通过 `hooks/` 目录提供 PyInstaller hook
- 最终生成 `dist/gpt2ref/gpt2ref.exe`

### 4. 打包产物

打包成功后输出目录：

```text
dist/gpt2ref/
```

关键文件：

```text
dist/gpt2ref/gpt2ref.exe
dist/gpt2ref/_internal/
```

其中 `_internal/` 里会包含打包后的依赖、权重和资源文件。

### 5. 已验证的 EXE 运行情况

已实际验证：

```bat
dist\gpt2ref\gpt2ref.exe seizure 20370.edf
dist\gpt2ref\gpt2ref.exe report 20370.edf
```

两条命令都可运行。

验证结论：

- `exe` 能读取 `_internal/fine_tuning/` 下的模型权重
- `seizure` 模块在 `exe` 中可正常推理
- `report` 模块在 `exe` 中可正常生成报告

### 6. 已知打包注意点

- 全量 `--clean` 打包耗时较长，依赖体积较大
- 打包日志里会出现一些 warning，但当前不影响出包
- 控制台中文输出仍可能乱码
- 如果后续补上 `ICA` 功能的正式支持，需要确认打包环境包含 `scikit-learn`

## 推荐使用方式

### 源码方式

优先在项目内 `.venv` 运行：

```bat
.venv\Scripts\python.exe comcode.py seizure 20370.edf
.venv\Scripts\python.exe comcode.py report 20370.edf
```

### EXE 方式

适合交付或无 Python 环境的场景：

```bat
dist\gpt2ref\gpt2ref.exe seizure 20370.edf
dist\gpt2ref\gpt2ref.exe report 20370.edf
```

## 仓库整理约定

建议纳入 git 的内容：

- Python 源码：`comcode.py`、`seizure.py`、`sleep.py`、`report.py`、`spikeNet.py`、`utils.py`、`model/`
- 打包与运行配置：`exe.spec`、`new.spec`、`build_exe_venv.bat`、`requirements.txt`、`hooks/`
- 文档：`README.md`、`代码操作与打包说明.txt`

默认不纳入 git 的内容：

- 本地环境与构建产物：`.venv/`、`build/`、`dist/`、`lightning_logs/`
- 测试数据与样例 EDF：`*.edf`
- 大模型权重与微调检查点：`*.ckpt`、`*.pth`、`*.safetensors`
- 压缩包与临时产物：`*.rar`

如果后续需要共享权重，建议使用单独的模型下载说明或发布页，而不是直接提交到 git。
