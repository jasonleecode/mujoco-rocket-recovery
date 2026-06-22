# 🚀 MuJoCo 火箭回收仿真 (Vector-Thrust Rocket Recovery)

在 [MuJoCo](https://mujoco.org/) 中仿真一枚**矢量推力火箭的垂直回收着陆**:火箭底部是一个
**两轴万向矢量喷嘴 (gimbaled TVC nozzle)**,发射台上有一个 **H 形着陆标识**,火箭在下降段
对准 H 标识完成精准软着陆。

控制分两个阶段:

| 阶段 | 高度 | 控制器 | 作用 |
|------|------|--------|------|
| **Stage 1 接近/减速** | `> 12 m` | **经典算法**(级联 PD 制导 + 姿态控制) | 把火箭飞到 H 标识正上方、刹住下降速度、保持竖直 |
| **Stage 2 末端着陆** | `≤ 12 m` | **MLP 神经网络**(行为克隆训练) | 精准坐到 H 标识上,实现柔和触台 |

闭环评测(50 次随机初始条件):**成功率 100%**,平均水平误差 **4.4 cm**,触台速度
**0.70 m/s**,着陆倾角 **0.44°**。

---

## 物理模型 (`models/rocket.xml`)

- **火箭本体**:细长圆柱 + 头锥,总长约 10 m,质量约 1055 kg,带自由关节(6 自由度)。
- **矢量喷嘴**:底部两个正交 hinge 关节组成万向架,推力沿喷嘴轴施加。偏转喷嘴会让推力
  线偏离质心,从而产生**控制力矩**——这正是真实火箭 TVC 的工作原理。
- **着陆腿**:4 条外撑腿,腿尖低于喷口,保证先于引擎着地。
- **发射台 + H 标识**:地面上的方台,顶部用 3 根白色条拼出字母 **H**,作为着陆目标。
- 推力上限 22 kN,推重比 ≈ 2.1;矢量偏转范围 ±12°。

碰撞分组(`contype/conaffinity`)让箭体各部件互不碰撞,只与地面/发射台碰撞,避免万向
喷嘴自碰撞。

## 控制算法

### Stage 1 — 经典级联控制 (`rocket_landing/controllers/classical.py`)

1. **制导外环**:对水平位置/速度做 PD,叠加一条随高度收敛的下降速度剖面,得到世界系下
   的期望比力 `f_des`(含重力补偿)。
2. **姿态内环**:火箭长轴应指向 `f_des`,姿态误差经 PD 生成期望体轴力矩。
3. **控制分配**:利用喷嘴相对质心的力臂,把期望力矩换算成万向偏转角;油门由 `|f_des|` 决定。

### Stage 2 — MLP 末端策略 (`rocket_landing/controllers/mlp.py`)

一个小型 MLP(`13 → 128 → 128 → 64 → 3`),输入 13 维观测,输出
`[油门, 万向x, 万向y]`(经 sigmoid/tanh 压到合法区间)。通过**行为克隆**经典专家在末端
阶段的轨迹训练得到,并注入探索噪声扩大状态覆盖(DAgger 式技巧),使策略对偏离标称轨迹
的状态更鲁棒。

两阶段切换带迟滞,见 `rocket_landing/guidance.py`。

### 观测向量 (13 维)

```
[0:3]  相对 H 标识的位置 (dx, dy, dz)
[3:6]  线速度 (vx, vy, vz)
[6:8]  箭体 +Z 轴在世界 XY 的倾斜分量 (lean_x, lean_y)
[8]    cos(倾角)  —— 1 表示完全竖直
[9:12] 角速度
[12]   高度(腿尖距台面)
```

---

## 安装

```bash
pip install -r requirements.txt   # mujoco, numpy, torch
```

## 使用

```bash
# 1) 可视化运行:仅经典控制器
python scripts/run_sim.py

# 2) 可视化运行:完整两阶段控制器(经典 → 已训练 MLP)
python scripts/run_sim.py --policy models/mlp_policy.pt

# 3) 无显示环境,直接打印结果
python scripts/run_sim.py --headless --policy models/mlp_policy.pt

# 4) 重新训练末端 MLP(行为克隆)
python scripts/train_mlp.py --episodes 300 --epochs 200

# 5) 批量评测
python scripts/evaluate.py --controller two-stage --policy models/mlp_policy.pt --episodes 100
python scripts/evaluate.py --controller classical --episodes 100

# 6) 测试
python -m pytest tests/ -q
```

仓库已附带训练好的 `models/mlp_policy.pt`,开箱即用。

## 项目结构

```
mujoco_roket/
├── models/rocket.xml              # MJCF:火箭 + 矢量喷嘴 + 发射台/H 标识
├── rocket_landing/
│   ├── env.py                     # 仿真环境封装(动作/观测/奖励)
│   ├── guidance.py                # 两阶段切换控制器
│   ├── rollout.py                 # 回合运行 / 评测
│   ├── utils.py                   # 四元数/旋转工具
│   └── controllers/
│       ├── classical.py           # Stage 1 经典级联控制
│       └── mlp.py                 # Stage 2 MLP 策略
├── scripts/
│   ├── run_sim.py                 # 可视化/无头运行
│   ├── train_mlp.py               # 行为克隆训练
│   └── evaluate.py                # 批量评测
├── tests/test_env.py
└── models/mlp_policy.pt           # 训练好的末端策略
```

## 后续可拓展

- 用强化学习(PPO/SAC)替代行为克隆,进一步优化末端策略(`env.py` 已提供 shaped reward)。
- 加入下视相机 + 视觉网络,从图像中检测 H 标识(当前用真值相对位姿模拟"对准")。
- 风扰、推力延迟、传感器噪声等域随机化以提升鲁棒性。

## License

MIT — 见 [LICENSE](LICENSE)。
