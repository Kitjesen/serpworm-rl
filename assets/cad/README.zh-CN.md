# 五体节机器人 CAD / URDF 参考包

这个目录收录当前本地已有的长虫 CAD 外观资源和 SOFA 装配参考文件。

## 内容

```text
meshes/longworm2/
├── longworm2.SLDASM.urdf
├── base_link.STL
├── front*_Link.STL
├── back*_Link.STL
└── w1-1_Link.STL ... w6-4_Link.STL

single_segment/
├── cad_assembly.json
├── cad_assembly.py
├── cad_assembly_static_audit.json
├── cad_assembly_validation.json
└── sofa_worm_env.py
```

## 与 v5 GPU 模型的关系

这些 URDF 和 STL 用于 CAD 外观、装配检查和历史 SOFA 场景。当前主线 GPU 动力学使用参数化输入模型：

```text
assets/inputs/
├── topology_5.npz
└── parameters.json
src/serpworm/
├── physics/gpu_core.py
└── envs/tensor_env.py
```

两条路线的体节数量、隔板、钢片和轮子结构需要保持对应。当前 CAD 网格尚未接入 GPU 碰撞求解器，不能用这个目录替代参数化拓扑输入。

## 查看方式

可以用 RViz、Blender、MeshLab 或支持 URDF 的机器人查看器加载：

```text
meshes/longworm2/longworm2.SLDASM.urdf
```

URDF 中的 STL 路径需要按查看器工作目录调整到 `meshes/longworm2/`。

## 来源和状态

源目录为 `worm-sim-ribbon-publish/meshes/longworm2` 和 `worm-sim-ribbon-publish/single_segment`。这是本地已有的 CAD/SOFA 参考资产，已随本交接提交上传。它们没有被宣称为已经完成标定的实物数字孪生。

