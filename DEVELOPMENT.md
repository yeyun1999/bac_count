# bac_count 开发说明（交接文档）

> 面向：接手本项目的开发者 / AI agent。写完即生效，与代码冲突处以代码为准。
> 日期：2026-10-24 | HEAD: `2240616` | 33 测试全绿

## 1. 项目是什么

培养皿菌落计数桌面工具（PySide6 + OpenCV）：导入平板照片 → 自动检测菌落并标注 →
调参实时重算 → 手动增删校正 → 批量导出结果。无摄像头，纯图片流程。

需求与里程碑的原始设计文档在 `PLAN.md`（已完成 M1–M7）；
`COLONY_GUI_TODO.md` 是会话进度笔记（含样例校准记录与 v1.1 待办）。

## 2. 快速开始

```bash
# 环境：uv（本机 0.11.x），Python 3.12（.python-version 锁定，uv 自动装）
uv sync                                  # 装依赖，含 dev 组
uv run bac-count sample/                 # 启动并加载样例文件夹
uv run python -m bac_count               # 等价启动（也可带路径参数）

QT_QPA_PLATFORM=offscreen uv run pytest -q   # 全量测试（33 个）
QT_QPA_PLATFORM=offscreen uv run pytest tests/test_ui_smoke.py::test_xxx -q  # 单测
```

依赖版本（uv.lock 已锁）：opencv-python-headless 5.0.0 / numpy 2.5.4 / PySide6 6.12.0。
dev 组：pytest、pytest-qt。用 headless 版 OpenCV 是刻意的——渲染全走 Qt，避免 X11 依赖冲突。

## 3. 架构与文件地图

```
src/bac_count/
├── __main__.py / __init__.py      # 入口 → ui.main_window.run()
├── core/                          # 纯逻辑层：零 Qt 依赖，可单测
│   ├── models.py (98)             # Colony/ExclRegion/EditState/DetectionResult/ParamSpec + merge()
│   ├── preprocess.py (106)        # load_image / make_dish_mask / prepare（预处理链）
│   ├── pipeline.py (191)          # run_detection 总调度 + PRE/POST_SCHEMA + default_params
│   ├── registry.py (46)           # Detector ABC + @register 注册表 + PrepContext
│   ├── detectors/{threshold,adaptive,blob,hough,hsv}.py   # 5 个算法，一文件一个
│   ├── edits.py (75)              # EditHistory：增删/undo/redo（纯逻辑）
│   ├── split.py (108)             # 粘连拆分：距离变换+分水岭（split_on 开关，默认关）
│   ├── color.py (86)              # 颜色过滤：菌落色保留/排除色剔除（color_on 开关，默认关）
│   ├── dedup.py (48)              # 重叠/包含去重：交集占小圆比例阈值（默认开）
│   ├── imio.py (36)              # Unicode 路径安全读写（imread_any/imwrite_any，绕 Windows 代码页坑）
│   └── export.py (75)             # draw_overlay / PNG / 汇总CSV / 完整JSON
├── ui/                            # Qt 层
│   ├── main_window.py (632)       # 组装一切：状态机、调度、编辑、导出、预设、取色接线
│   ├── canvas.py (536)            # 画布：缩放/平移/标注/交互/视图切换/取色模式+放大镜(_LoupeWidget)
│   ├── param_panel.py (332)       # schema→控件自动生成 + 折叠分组 + tooltip + ColorListWidget
│   ├── file_list.py (161)         # 左侧文件列表：拖放/多选/右键菜单
│   ├── worker.py (56)             # DetectWorker(QThread) + 预处理图回传
│   └── presets.py (26)            # 参数预设 JSON → ~/.bac_count/presets/
tests/                             # 75 个：算法/编辑逻辑/GUI 冒烟+交互回归
sample/                            # 5 张真实样例（RGBA PNG ~800px，见 §6）
```

数据流（一次调参）：

```
滑杆 valueChanged → MainWindow._on_param_changed → params[key] + _commit_params()
  → 150ms debounce → request_detection()（busy 则置 pending，完成后合并补发）
  → worker.submit(seq, bgr, alpha, params)   [QueuedConnection → 工作线程]
  → run_detection → done(seq, colonies, prep_bgr)  [回 GUI 线程]
  → seq 守卫丢弃过期 → last_auto → _refresh_merge()
  → merge(auto, edits) → canvas 重绘 + 状态栏计数
```

## 4. 坐标系约定（改 UI/算法前必读）

- **`Colony` 的 x/y/r/area 永远是原图坐标**。所有持久化（image_params、edits、导出）都用原图坐标。
- `run_detection` 内部把图缩到长边 ≤ `WORKING_MAX=1200` 跑检测，返回前把结果乘回 `1/scale`。
  返回值 `(colonies_原图, ctx_工作尺度, scale)`，`scale = 工作/原图`。
- **画布场景坐标 = 工作图坐标**：绘制时乘 `scale`（`canvas.set_colonies(cols, scale)`），
  点击回传场景坐标，由 `MainWindow._on_add_click` 除以 `scale` 转回原图坐标。
- 样例图都 ≤1200px（scale=1），改缩放逻辑时记得用大图测试换算路径。

## 5. 核心语义

### 5.1 检测结果合并（models.merge）
```
最终结果 = (自动检测 − 排除区) ∪ 手动添加
```
- 删除一个自动菌落 = 记录 `ExclRegion(x,y,r)`；重检测后，与新候选满足
  `dist(c, e) <= c.r + 0.5*e.r` 的被剔除 → **调参不丢人工校正**。
- 删除手动菌落 = 直接从 additions 移除。
- 合并后统一重编号 id（1..N）。`DetectionResult` 带 auto_total/kept/removed/added 供状态栏。
- undo/redo：`EditHistory` 每次编辑前存 EditState 深拷贝快照（上限 100），按图片路径各一份。

### 5.2 每图独立参数（main_window.image_params）
- `self.params` = 当前显示图的参数；每次改动 `_commit_params()` 写回 `image_params[str(path)]`。
- 切图（`load_current`）：该图有快照 → 恢复并重建面板；没有 → **继承当前参数**（= 上一张的）。
- 复制：文件列表右键「复制当前图参数 → 选中项/全部其他图」只写快照，导航过去时生效。
- 批量导出时未访问过的图用 `default_params()` 重算（无继承链可言）。

### 5.3 线程模型（worker.py）
- `DetectWorker(QObject)` moveToThread；`submit`/`done` 都是信号（自动队列连接）。
- 并发控制全在 GUI 侧：`_busy` + `_pending`（合并请求）+ `_seq`（过期结果守卫，
  同时防跨图竞态：切图后旧检测回包 seq < _seq 被丢弃）。
- `done` 携带 `(seq, colonies, prep_display)`；prep 是工作尺度 BGR 预处理图+皿ROI红圈，
  在工作线程生成（`_prep_display`），UI 只转 pixmap。

### 5.4 预处理链（preprocess.prepare，顺序固定）
```
通道选择(gray/r/g/b/h/s/v) → 皿外填充皿内中值(关键!) → 阴影校正(减大核中值)
→ 高斯模糊 → CLAHE → 极性统一(输出恒为"菌落暗")
```
极性统一后所有算法只处理暗色目标（SimpleBlobDetector 固定 blobColor=0）。

### 5.5 培养皿 ROI（make_dish_mask）
- 优先 alpha 通道（样例图皿外透明）→ 最大轮廓 minEnclosingCircle → 半径 ×(1−rim_pct)。
- 无 alpha 时 HoughCircles 找大圆兜底；再失败返回 None（全图可检测）。
- 后处理要求菌落中心落在 mask 内。

### 5.6 粘连拆分（split.py 108 行，默认关，pipeline 在检测后、颜色过滤前调用）
纯距离变换 + 分水岭，无碎片守卫（守卫版历史见 git `d5cf6c7`）：

- **前景二值图**：adaptiveThreshold（复用面板 block/C）→ **按外轮廓回填**。
  回填是必须的：自适应阈值只留"比局部均值暗 C"的边带，比 block/2 大的菌落中间成空洞，
  dt 变环状 → 处处假峰（实测 12+9 粘连对在回填前拆不出、回填后正常）。
- **触发**（工作坐标）：`area ≥ 全体中位面积 × split_factor`（默认 1.6）。
  中位数含大量 min_area 以下的碎点，P72 实测中位=3px² → 阈值 5px²，96 个候选里 45 个进拆分线；
  这是已知缺陷（拆分线太低会浪费算力，但误拆由"单峰不拆"挡住，故暂未修）。
- **找核**：`cd ≥ dilate(cd, 半径 sep)` 的**局部极大平台**（sep = 0.5×dt 峰值），
  平台连通域数 = 菌落数；`split_peak`（默认 0.55）只丢掉强度低于 `峰值×split_peak` 的弱核。
  不用"全局峰值比例一刀切"：粘连越紧鞍部越浅，一刀切要么拆不开、要么 N 联团只拆成 2 块。
- **分水岭**：标记 1=背景、2..=各核、0=前景未知；盆地即子菌落，
  面积 < `15%×原候选` 的碎屑丢弃，**有效碎片 <2 时整体作废、保留原候选**
  （否则碎片过不了下游 min_area，等于把菌落拆没——E1-2 针尖菌落 45→43 事故）。
- **物理极限**：小菌落贴在大菌落上时其中心不是 dt 局部极大（12+6 实测拆不动）；
  中心距 < 半径（重叠过半）同样拆不动——都靠手动。
- **已知假阳性（守卫删掉的代价）**：马克笔笔画被回填后是实心长条，粗段/交叉处能出核，
  拆出 2–3 个假菌落/图（E12 (484,429)(504,567)、P72 (183,266) 实测）。
- 样例基线（adaptive 默认参数，开/关拆分）：E0-3 51→52、E12 5→9、P24-2 110→111、
  P24-3 109→114、P72 22→23。逐块核对：E0-3(120,284)+(129,280)、E12(170,187)+(210,201)、
  P24 若干对/链均为**真粘连**，E12+P72 的 3 个落在笔画上。

### 5.7 颜色过滤（color.py，默认关，pipeline 在拆分后、几何过滤前调用）
- 两张取色表：`keep_colors`（菌落色，非空时与所有表色 Lab 距离 > `keep_tol` 的候选去掉）、
  `drop_colors`（培养基/墨色等，距离 < `drop_tol` 去掉）。表为空 = 该规则不生效。
- 距离在 cv2 8bit Lab：同色深浅差 <20，红↔黑/绿 >100（已实测标定，默认 60/35 就来自这）。
- 取色锚定：**中心像素做种子**，内切圆内只统计与种子色距 ≤45 的像素再取中位——
  笔画状候选的圆内 95% 是培养基，直接中位会得到培养基色（踩坑 §8-16）。
- 放在拆分后（2026-10-25 用户要求调整，原为拆分前）：分水岭拆出的子碎片也参与颜色判定；
  代价是拆分的中位面积基准包含异色大块（笔画太薄 dt<3 会被拆分支线跳过，实测无影响）。
- UI 取色器：面板色块行 🎨 → 画布进入取色模式（左键取一个自动退出/自动开 color_on，
  Ctrl+左键连取，右键/Esc 取消，**按住空格=临时平移**（取色暂停、放大镜隐藏））；
  取色时光标旁跟随**圆形放大镜**（_LoupeWidget：源区域随视图缩放自适应，
  观感恒为**屏幕当前倍率×3**（`LOUPE_REL`，源区域=D/(view_scale×3)，钳制 4~160 场景 px），
  最近邻放大+十字+RGB 色卡，事件驱动零定时器；WA_TransparentForMouseEvents 不挡点击；
  滚轮缩放后原地刷新（记住 `_pick_pos`）；换图自动 cancel_pick）；
  色块点击=移除该色；`ColorListWidget` 值永远整表替换
  （配合每图快照 `dict()` 浅拷贝是安全的——禁止原地 append）。
- E0-3 实测：常规参数下 keep/drop 均 51→51 零误伤；放宽几何（min_circ=0）63→59 正确滤出 4 个墨色笔画。

### 5.8 重叠去重（dedup.py，默认开，run_detection 在几何过滤后执行）
- 规则：交集面积/小圆面积 ≥ `dedup_ovl`（默认 0.6）→ 同一菌落，**贪心保留面积大的**。
  大圆套小圆（Hough 对深色中心菌落出双圈）是重叠比=1 的特例，同一参数覆盖。
- 安全性：相切≈0、中心距=半径≈0.39、中心距 1.8r≈0.1——相邻真菌落不会被误并；
  adaptive/blob 5 样例开/关计数完全不变（回归测试锁定），Hough 30→22、111→104 即用户报的双圈。
- 参数在「后处理过滤」组：`dedup_on` + `dedup_ovl`（0.3~0.95）。

### 5.9 HSV 取色模式（detectors/hsv.py，替代旧 H/S/V 六数值框）
- `hsv_colors`（colorlist，复用 ColorListWidget+🎨）+ `hsv_tol`（容差）+ `morph_k`。空表=返回 []。
- 每个取到颜色 → cv2 HSV 窗口：H ±tol×0.4（跨 0/179 自动拆两段回绕）、
  **S 向高饱和放满 / 向培养基只放一半，V 向暗放满 / 向亮只放一半**（菌落通常更饱和更暗；
  对称窗口在 tol=40 即吞掉整个培养基 → 一整片巨轮廓被 max_area 滤光 → 计数骤降 0，实测教训）。
- 存在意义：唯一在**原色图**上做区域分割的算法，灰度无反差时是唯一出路；
  颜色过滤只能删候选、不能发现候选，二者不可互替（用户曾质疑可否删除——结论：保留）。

## 6. 样例数据与算法基线

`sample/` 5 张 RGBA PNG（~800×800，皿外 alpha=0 且 **RGB 为纯黑**）：
浅褐色圆菌落（直径 8–25px）+ **黑色马克笔手写字**（最大干扰源，细长笔画+环状"0"）。
人工计数估计（GT）：E0-3≈50，E12≈7，P24-2≈110，P24-3≈110，P72≈25。

默认参数实测（`run_detection(bgr, alpha, default_params(algo))`）：

| 算法 | E0-3 | E12 | P24-2 | P24-3 | P72 | 备注 |
|---|---|---|---|---|---|---|
| **adaptive（默认）** | 51 | 5 | 110 | 109 | 22 | 综合最优，5 张手写字零误检 |
| blob | 51 | 1 | 109 | 106 | 23 | E12 低对比度大菌落漏检 |
| threshold | 0 | 2 | 82 | 80 | 1 | 低对比度下 Otsu 失效，仅高对比图可用 |
| hsv | 54 | — | — | — | — | 取色模式：默认 hsv_colors=[] → 全 0（不检测）；E0-3 采 2 色 tol=40 实测 54 |
| hough | 22 | 9 | 14 | 104 | 18 | 参数敏感易把笔画弧当圆；dedup 把双圈合并后计数更准 |

关键默认值（都是拿样例调出来的，改前先跑 `test_default_algo_regression_on_samples`）：
`rim_pct=0.12`（0.08 时皿缘暗环与手写字连成 40 万 px² 的巨环）、blob 圆度/惯性比 0.3/0.2、
HSV 取色窗口非对称（S 向高饱和放满、向培养基只放一半，V 反之——对称 ±tol 到 40 就吞培养基，见 §5.7）。

## 7. UI 交互一览

工具栏（11 控件）：`📂打开▾ | ◀ ▶ | 🗂 | 浏览/添加/删除 | ↶ ↷ | 📤导出▾ ⚙预设▾`

| 快捷键 | 功能 |
|---|---|
| 1 / 2 / 3 | 浏览 / 添加 / 删除模式 |
| 按住空格 | 任意模式临时变平移（手形） |
| 按住 Ctrl | 添加↔删除临时互换（光标同步提示） |
| 滚轮 / 中键拖拽 / 0 | 缩放 / 平移 / 适应窗口（Ctrl± 也在，无按钮） |
| Alt+←/→ | 上一张/下一张 |
| Ctrl+Z / Ctrl+Y | 撤销/重做（仅编辑操作） |
| Ctrl+E | 文件列表 dock 折叠 |
| 列表内 Delete | 移除选中（不删磁盘） |

- 画布**左上角**悬浮两态：**原图 / 预处理**（`Canvas.set_view_mode`）；**右上角**「显示标记」复选框
  （`Canvas.set_show_marks`）独立控制菌落圈/编号叠加的显隐。二者解耦：预处理视图下也能看标记，
  便于对照灰度产物与检测结果。预处理图皿 ROI 红圈已烘焙进图，标记叠加在其上。
- 文件列表手势约定（§8.9/8.10 的血泪）：**无修饰单击/双击/Enter = 切图**；
  Ctrl/Shift 多选不切图；右键未选中项 = 加选；右键菜单在打开瞬间快照选区并显示数量，
  目标数为 0 的菜单项置灰。当前显示图**加粗**。
- 参数面板：按**检测执行顺序**分七组可折叠——皿 ROI → 预处理 → 检测算法（选择+算法参数同组）→
  粘连拆分 → 颜色过滤 → 后处理过滤 → 手动校正（默认收起 预处理/后处理过滤/手动校正），折叠态存 QSettings
  （org/app 均为 `bac_count`，Linux 落 `~/.config/bac_count/bac_count.conf`）。
  每个参数标签和控件都有 tooltip 解释（`ParamSpec.desc`，新增参数必须写）。
- 导出（对**全部列表图**批量，进度框可取消）：PNG→目标目录 `annotated/`；
  CSV 一图一行（文件名/数量/自动/手加/手删，utf-8-sig）；JSON 单文件含每图
  path/params/colonies/edits。当前图用内存态，其余按各自参数快照即时重算。

## 8. 踩坑清单（务必读，全是本项目真实踩过并验证的）

**OpenCV / 数据**
1. RGBA 透明区 RGB=(0,0,0)：不处理会毒死 Otsu/adaptive（阈值被黑背景+黑字拉走）。
   修在 `prepare()`：皿外用皿内中值填充。
2. 皿缘暗环与马克笔字连通 → 巨大轮廓假阳性。`rim_pct` 默认 0.12 起。
3. OpenCV 5.0：`findContours` 返回 `(contours, hierarchy)`（无旧版 3 元组问题）。

**PySide6 6.12 行为**
4. `QListWidgetItem.row()` **不存在**（C++ 侧 protected）。取行必须 `listWidget.row(item)`。
   曾导致右键菜单动作静默失败——PySide6 把虚函数覆写里的异常打到 stderr，GUI 用户看不见。
   ⇒ **给菜单/槽逻辑写 e2e 测试，别只测 happy path**。
5. `QToolButton` 无视样式表 `text-align:left`（实测文字仍居中）→ 分组标题用自定义
   `_HeaderWidget`（QLabel+stretch 的点击行）。
6. `QSettings.setPath(IniFormat, UserScope, tmp)` 不生效 → 测试曾写进真实 `~/.config`。
   ⇒ 隔离方式是**构造注入**：`ParamPanel(settings=...)` / `MainWindow(settings=...)`，
   测试传 `QSettings(str(tmp_path/"x.conf"), QSettings.Format.IniFormat)`。
7. 枚举带作用域名：`QContextMenuEvent.Reason.Mouse`（不是 `TriggerByMouse`）；
   `QMouseEvent.pos()` deprecated → `ev.position().toPoint()`。
8. `QMenu(title, parent)` 的 parent 必须是 QWidget，不能传 QAction。
9. 列表导航若挂 `currentItemChanged` + 每次 `clear()` 重建 → Ctrl/Shift/右键全被销毁选区。
   ⇒ 导航只挂显式手势（自算 `mouseReleaseEvent` 判 modifiers），列表只在结构变化时重建。
10. Qt 默认右键未选中项会清掉多选 → `mousePressEvent` 里改成 `setSelected(True)` 加选。

**测试基建**
11. QTest 合成鼠标事件**不会生成 QContextMenuEvent** → 测试里手动
    `QApplication.sendEvent(list.viewport(), QContextMenuEvent(...))`（注意发 viewport，
    QAbstractScrollArea 从 viewport 转发）。
12. `qtbot.mouseClick` 的 modifiers 只能位置参数：`qtbot.mouseClick(vp, Qt.LeftButton, Qt.ControlModifier, pt)`。
13. 测右键菜单：monkeypatch `FileList._popup_menu` 捕获 QMenu 再 `act.trigger()`，
    否则 `menu.exec()` 阻塞测试。
14. GUI 测试等检测完成用 `with qtbot.waitSignal(w.service.worker.done, timeout=20000)`，
    包在触发动作外面；断言前 `qtbot.wait(300)` 等 merge 落画布。

**颜色采样 / 控件重建（2026-10 颜色过滤时新增）**
15. 重建含永驻按钮的 QHBoxLayout：`while count(): takeAt(0)` 会把 🎨/✕ 一起 deleteLater。
    异步删除构造期不炸，下次 `_rebuild()` 再引用 self._btn_clear 才崩 "already deleted"。
    ⇒ 自己维护动态子控件列表（`self._sw_btns`），只删它们、`insertWidget(i)` 保持在前部。
16. 笔画/细长线状候选：0.6r 内切圆几乎全是背景像素，"中位色"测出来是培养基色而不是墨色
    （E0-3 实测距黑 130，drop 黑色完全滤不掉手写字）。⇒ 中心像素做种子色聚类（§5.7）。
17. **切算法不补默认参数 → 检测线程 KeyError**（用户实测 HSV 崩溃）：`default_params(algo)`
    只含当前算法的键，GUI 里切算法只改 `params["algorithm"]`，新算法的键全靠面板控件
    "被摸过才写入"。⇒ `main_window._on_param_changed("algorithm")` 里 setdefault 合并
    `get_detector(name).schema()` 的默认值；detector 侧读取一律 `params.get(key, 兜底)`。
18. HSV 取色窗口对称 ±tol 是陷阱：菌落↔培养基 S/V 只差 35~40，tol=40 整盘培养基入掩膜，
    RETR_EXTERNAL 只剩一个巨轮廓被 max_area 滤光 → 计数从 54 直接跳 0（无渐变）。
    ⇒ 非对称窗口向培养基方向半格（§5.9）。另：OpenCV 5 的 `inRange` 拒收 np.int64 标量元组，
    边界值必须转 Python int。
19. **Windows 中文路径**：`cv2.imread/imwrite` 按进程 ANSI 代码页解析路径，UTF-8 中文被
    GBK 误解码（课题→璇鹃）→ 用户真实数据目录直接读不出（v1.0.0 exe 实报）。
    ⇒ 一律走 `core/imio.py`：Python `read_bytes`（宽字符 API）+ `cv2.imdecode`，
    写侧 `cv2.imencode` + `write_bytes`。CSV/JSON/预设本来就走 Python IO，不受影响。
20. **面板按钮抢键盘焦点 → 画布快捷键全哑**：点 🎨 后焦点停在按钮上，空格/Ctrl/Esc
    发给按钮而非画布（用户实测：取色中空格不平移，"Ctrl 取色后再按空格才生效"——因为
    点击画布才把焦点带回来）。⇒ `enter_pick()` 里 `setFocus()` + ColorListWidget 全部
    按钮 `NoFocus`。**QTest.keyPress(widget) 直接发给指定 widget、绕过焦点路由，这种 bug
    行为测试测不出**——要断言 `canvas.hasFocus()`（需 w.show()+waitExposed）。
21. **覆写 `mouseMoveEvent` 提前 return 不调基类 → 滚轮缩放锚点漂移**：
    `AnchorUnderMouse` 靠基类在 move 事件里记录的鼠标位置；取色分支吞掉 move 后该位置
    过期，wheelEvent 缩放中心乱飘（浏览模式正常，因为走了 super()）。⇒ 取色分支也先
    `super().mouseMoveEvent(ev)` 再刷新放大镜。
22. **做"验证测试能否抓到 bug"的破坏性实验时，别用 `git checkout -- file` 回滚**——
    会把该文件所有未提交改动一起灭掉（本项目实际烧过一次全部重打）。用临时副本或先提交。

**无 alpha 通道图片（2026-10-25 用户实报，Windows + 手机照片）**
23. `load_image()` 对 JPG/BMP 返回 `alpha=None`，而 `load_current()` 只在 `scale < 1.0`
    （原图 > WORKING_MAX=1200）的分支里无条件 `cv2.resize(self.alpha, ...)` →
    OpenCV `(-215:Assertion failed) !ssize.empty()` 直接崩，小 JPG 反而正常（不走缩放分支）。
    ⇒ 缩放 alpha 前判空；回归测试 `test_load_large_jpg_without_alpha`（造 WORKING_MAX+400 的
    JPG 走真实 load_paths）。凡"样例全是带 alpha 的 PNG"的地方都要想一遍无 alpha 的路径。

## 9. 如何扩展

### 加一个检测算法（唯一常见扩展点）
```python
# src/bac_count/core/detectors/mine.py
from bac_count.core.registry import Detector, PrepContext, register
from bac_count.core.models import ParamSpec
from bac_count.core.pipeline import contours_to_colonies

@register
class Mine(Detector):
    name = "mine"; display = "我的算法"
    def schema(self): return [ParamSpec("k","参数", "int", 5, group="算法参数",
                                        min=0, max=99, desc="悬浮解释必填")]
    def detect(self, ctx: PrepContext, params) -> list[Colony]:
        ...  # ctx.gray 已保证"菌落为暗色"；返回工作图坐标，后处理由 pipeline 统一过滤
```
再到 `core/detectors/__init__.py` 加一行 import 即完成注册（UI 下拉、预设、测试自动继承）。
面积/圆度/长宽比过滤**不要**在算法里做，交给 `_post_filter`。

### 加一个面板参数
在对应 schema 列表加 `ParamSpec`（**desc 必填**，有测试兜底），控件自动生成；
若不希望改动触发重检测，参考 `_on_param_changed` 里 `manual_r` 的短路写法。

## 10. 已知局限 / 建议方向（未做，见 COLONY_GUI_TODO）

- E12 类低对比度大菌落：adaptive 漏 ~2/7。候选：默认开 shading 校正或 CLAHE。
- 粘连拆分已做（§5.6），但重叠 >50% 的极端粘连仍是盲区：候选方向 DL 分割（`registry` 即扩展点）。
- 排除区（手动删除的圆）在画布上不可见，误删难察觉——值得先做。
- 批量无交互处理、CFU=计数×稀释倍数、多语言：均 P2 未动。
- 检测在工作分辨率(≤1200px)跑，导出时 `_result_for_path` 也是同样缩放——
  若要求导出精度与显示完全一致，需加"原图全分辨率重算"路径。

## 11. 仓库约定

- git：main 直提，提交信息中文、`type: 摘要`（feat/fix/refactor/docs）；
- CI（.github/workflows）：`main.yml`=push 跑 pytest（ubuntu/offscreen）；`build-exe.yml`=Nuitka
  单文件 exe 构建（Actions 里可手动触发，也被 release 复用，改构建参数只动这一处）；
  `release.yml`=推 `v*` 标签→调 build-exe→传 exe 到 GitHub Release。exe 版本号取自 pyproject
  （importlib.metadata），图标源文件：`src/bac_count/resources/logo.png`。
  `*_TODO.md` 在 .gitignore（会话笔记不入库）。
- 代码风格：core 层零 Qt、纯函数优先；UI 层信号驱动；行宽 ~110；中文注释/文案。
- 每次功能改动必须：① 补/改测试（GUI 行为要真实事件模拟）② 跑 offscreen 全量
  ③ 同步 README 操作表。视觉类改动用 `widget.grab()` 离屏截图 + 像素扫描验证
  （本项目用这法测过对齐、标注正确性）。
