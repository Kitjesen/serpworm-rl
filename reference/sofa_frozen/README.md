# 冻结的 SOFA 对照来源

这些文件来自原项目 `single_segment/physics_core_20261003/v1/source` 与拓扑导出脚本，保留原状供追溯。它们不是本仓库的训练入口，也不代表最新的物理修正。

`export_reference.original.py` 保留了原项目目录布局；在此目录直接运行需要先调整路径并准备相应 SOFA、BeamAdapter、矩形截面扩展及 Python 依赖。当前仓库训练直接使用 `inputs/` 的已导出拓扑，不需要执行它。

保留原文件而非静默改写它，是为了区分“历史 SOFA 对照”和“新的独立 GPU 实现”。
