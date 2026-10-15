# player-highlights

从篮球比赛录像中，按**球衣号码 + 球衣颜色**自动找出一名球员，产出：

- 🎬 **精华片段**：球员持球、触球、投篮的片段，可选 1× 原版或 1.2 / 1.6 / 2.0 倍跟拍放大，每段带起止时间、走动时钟、转场，结尾附统计卡
- 📼 **完整在场版**：球员每一段在场时间
- ⏱️ **上场时间**：支持停表 / running clock、节 / 半场
- ✂️ **片段清单**：可用一行文字调整、删除或补加片段

全部在本机 Apple Silicon Mac 上运行，**不上传任何画面**。

Find one basketball player by jersey number and colour in game videos and get a highlight reel, a full on-court cut
and minutes played — locally on an Apple Silicon Mac. English usage: see the skill file
[`.agents/skills/player-highlights/SKILL.md`](.agents/skills/player-highlights/SKILL.md).

## 需要什么

- Apple Silicon Mac（M 系列芯片，M1 及之后），建议 16GB 内存
- 约 15GB 硬盘放模型；处理时每场比赛临时约 6GB（跑完自动清除）
- `brew install ffmpeg uv git`
- 比赛录像：固定机位或手持跟拍都可以，每节 / 每半场一个文件，或一个全场文件

## 安装

```bash
git clone https://github.com/colakang/player-highlights.git
cd player-highlights
./setup.sh              # 安装环境、EdgeTAM、检测模型（不需要任何 API key）
./setup.sh --prefetch   # 可选：预先下载 Qwen2.5-VL 视觉模型（约 7GB）
```

## 用法一：交给 AI 助手（推荐）

在仓库目录里打开 **Claude Code** 或 **Codex**，直接说：

> 帮我把 ~/Movies/0920 这场比赛做成 12 号的精华，白色球衣

助手会按 skill（`.agents/skills/player-highlights/SKILL.md`）逐项确认：每个文件是第几节、赛制和计时方式、号码、颜色、放大倍数、输出位置，然后在后台运行并汇报结果。

## 用法二：命令行

```bash
.venv/bin/python pf.py \
  --videos ~/Movies/0920 \                 # 文件夹（按文件名排序）或逐个文件
  --periods Q1,Q2,Q3,Q4 \                  # 每个文件对应的节次：Q1..Q4 / H1,H2 / FULL
  --format quarters:8:stop \               # 赛制：quarters|halves|full : 每节分钟 : stop|running
  --number 12 --color white --name 12_ALEX \
  --zooms 1,1.6                            # 1 = 原版；可多选 1.2 / 1.6 / 2.0
```

输出默认在 `~/Movies/player-highlights/<比赛文件夹名>/`（`--out` 可改到任何位置，包括 NAS）：

| 文件 | 内容 |
|---|---|
| `12_ALEX_highlight.mp4` | 原版精华 |
| `12_ALEX_highlight_zoom1.6.mp4` | 1.6 倍跟拍精华 |
| `*_clips.txt` | 片段清单 |
| `12_ALEX_full_game.mp4` | 完整在场版 |

每 15 分钟录像约需 40–60 分钟处理；中途中断后用同一条命令重跑会从断点继续。

### 调整片段

直接编辑 `_clips.txt`（每行一段，开头是节次和开始时间，后面的 `+0 +0` 是开始 / 结束的调整秒数）：

```
Q3 09:41 -2 +3    # 开始提前 2 秒、结束延后 3 秒
Q2 05:10 x        # 删掉这段
Q4 10:20 10:35 +  # 新增一行：补加一段
```

然后用原命令加 `--adjust <这个文件>` 重跑，只会重新剪辑。

## 准确度（如实说明）

在作者的 5 场青少年比赛（1080p 场边机位）上：

- 上场时段：与人工核对相比，召回和精确度大多在 93–100%
- 认出球员的时间：在场时间的 65–85%
- 自动精华：覆盖球员约 3/4 的持球回合；远端半场、抢篮板的人堆、无球防守最容易漏

画质是最大因素：**4K、快门 1/250 秒以上、机位高且居中**会明显更准。

## 工作原理（简述）

检测（YOLO11s，CoreML）→ 跟踪（McByte / BoT-SORT + OSNet）→ 号码识别（Qwen2.5-VL 3B / 7B，MLX）→ 哨声 / 画面分段 + HMM 判断上场时段 → 阵容约束分配身份 → 以读到号码的帧为锚点用 EdgeTAM 跟住球员 → 放大检球 + Qwen 复核触球 → 投篮候选 → 硬件编码剪辑。详见 [AGENTS.md](AGENTS.md)。

## 参与

欢迎提 Issue 和 PR。请勿提交任何比赛画面、截图或标注数据（都是未成年人）。改进准确度的 PR 请附上测量方法和前后数字。

## 许可

[AGPL-3.0](LICENSE)。第三方组件和模型许可见 [NOTICE.md](NOTICE.md)。默认检测模型用 Roboflow Universe 公开数据集（CC BY 4.0）训练。
