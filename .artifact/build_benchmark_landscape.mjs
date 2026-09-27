import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/miker/Documents/ChatGPT/rsi/Env-Agent-rsi/outputs/env-evolution-research";
const previewDir = "/tmp/env-agent-rsi-benchmark-previews";
const outputPath = `${outputDir}/benchmark_landscape.xlsx`;
const font = "Arial";

const sources = [
  ["S00", "Env-Agent-RSI project seed design", 2026, "", "本项目的最小状态化 API 环境方案。"],
  ["S01", "EnvHarness: Awakening Static Worlds for Agent Learning", 2026, "https://arxiv.org/abs/2608.19880", "Stage/Contract/Chain、五个 benchmark、EnvRigger 与限制。"],
  ["S02", "google-research/envharness", 2026, "https://github.com/google-research/envharness", "当前代码中的 Setup/Rules/Link、toy24 与 Bridge 接口。"],
  ["S03", "AppWorld", 2024, "https://arxiv.org/abs/2407.18901", "9 个应用、457 个 API、750 个任务与状态单测 evaluator。"],
  ["S04", "τ-bench", 2024, "https://arxiv.org/abs/2406.12045", "工具-Agent-用户交互；终态数据库比较与 pass^k。"],
  ["S05", "ALFWorld", 2020, "https://arxiv.org/abs/2010.03768", "文本具身环境；EnvHarness 的核心实验环境之一。"],
  ["S06", "WebShop", 2022, "https://arxiv.org/abs/2207.01206", "模拟电商网站；EnvHarness 的 RL 分析环境。"],
  ["S07", "WebArena", 2024, "https://webarena.dev/", "可执行网站环境与任务级 evaluator。"],
  ["S08", "WorkArena and BrowserGym", 2024, "https://arxiv.org/abs/2403.07718", "ServiceNow 知识工作任务与 BrowserGym 环境。"],
  ["S09", "SWE-bench", 2023, "https://arxiv.org/abs/2310.06770", "真实 GitHub issue、容器化代码库与测试验证。"],
  ["S10", "OfficeQA", 2025, "https://github.com/databricks/officeqa", "面向真实财务文档的 grounded reasoning benchmark。"],
  ["S11", "SpreadsheetBench", 2024, "https://spreadsheetbench.github.io/", "真实复杂电子表格操作与工作流评测。"],
  ["S12", "AgentBench", 2023, "https://arxiv.org/abs/2308.03688", "八类交互环境的通用 Agent 评测。"],
  ["S13", "AgentBoard", 2024, "https://arxiv.org/abs/2401.13178", "多轮 Agent 分析框架与 progress rate。"],
  ["S14", "OSWorld", 2024, "https://arxiv.org/abs/2404.07972", "369 个真实电脑任务、初始化与执行式 evaluator。"],
  ["S15", "AndroidWorld", 2024, "https://arxiv.org/abs/2405.14573", "116 个参数化移动任务，含 setup/success/teardown。"],
  ["S16", "Terminal-Bench 2.0", 2026, "https://arxiv.org/abs/2601.11868", "89 个真实终端任务、独立环境与综合测试。"],
  ["S17", "Continual Harness", 2026, "https://arxiv.org/abs/2605.09998", "Pokemon Red/Emerald 的不重置长程 harness 适应。"],
  ["S18", "Prime Agent", 2026, "https://arxiv.org/abs/2608.23552", "长程 harness；ARC-AGI-3、Factorio 等环境。"],
  ["S19", "Darwin Gödel Machine", 2025, "https://arxiv.org/abs/2505.22954", "SWE-bench 与 Polyglot 上的开放式 Agent 代码进化。"],
  ["S20", "A Self-Improving Coding Agent", 2025, "https://arxiv.org/abs/2504.15228", "SWE-bench Verified、LiveCodeBench 与合成 Agent 任务。"],
  ["S21", "AFlow", 2024, "https://arxiv.org/abs/2410.10762", "MCTS 搜索 workflow；六个 QA/代码/数学 benchmark。"],
  ["S22", "SkillsBench", 2026, "https://arxiv.org/abs/2602.12670", "86 个任务、11 个领域、确定性 verifier。"],
  ["S23", "PAIRED / Unsupervised Environment Design", 2020, "https://papers.nips.cc/paper/2020/hash/985e9a46e10005356bbaf194249f6856-Abstract.html", "regret 驱动的困难但可解环境课程。"],
  ["S24", "ACCEL", 2022, "https://proceedings.mlr.press/v162/parker-holder22a.html", "编辑已有 level 并用 regret 保持能力前沿。"],
  ["S25", "Procgen Benchmark", 2019, "https://arxiv.org/abs/1912.01588", "16 个程序化游戏环境与未见关卡泛化。"],
  ["S26", "Enhanced POET", 2020, "https://proceedings.mlr.press/v119/wang20l.html", "环境-策略配对种群、novelty、迁移与开放式档案。"],
  ["S27", "OMNI-EPIC", 2024, "https://arxiv.org/abs/2405.15568", "用基础模型生成环境代码和奖励代码。"],
  ["S28", "EnvGen", 2024, "https://arxiv.org/abs/2403.12014", "LLM 生成和调整具身环境配置。"],
  ["S29", "GenEnv", 2025, "https://arxiv.org/abs/2512.19682", "难度对齐的生成式环境模拟器。"],
  ["S30", "Agent-World", 2026, "https://arxiv.org/abs/2604.18292", "从工具生态与数据库合成可执行环境和任务。"],
  ["S31", "VeriEnv", 2026, "https://arxiv.org/abs/2603.10505", "克隆可执行网站并用后端状态做确定性验证。"],
  ["S32", "BrowserGym", 2025, "https://github.com/ServiceNow/BrowserGym", "统一接入 MiniWoB、WebArena、WorkArena 等网页 benchmark。"],
  ["S33", "LiveCodeBench", 2024, "https://arxiv.org/abs/2403.07974", "持续更新、污染控制的代码评测。"],
];

// Scores are project-specific: control, verifier, reset, mutation fit, low setup cost, project fit.
const benchmarks = [
  ["Project seed: Stateful API micro-world", "项目起点", "本报告", "状态化 API", "Exactly-once append；区分提交前/后失败", "终态 DB + 审计日志", "完整快照", "初始数据、故障时点、观察、预算", 2,2,2,2,2,2, "最便宜且直接命中重复调用/幂等恢复；作为所有验证门禁的基准。", "S00"],
  ["EnvHarness toy24", "EnvHarness", "EnvHarness", "内存 toy env", "运行官方 smoke，验证 wrapper 与持久化", "内置确定性评分", "完整保存/恢复", "Setup/Rules/Link", 2,2,2,2,2,1, "先验证对 EnvHarness 语义的理解，再实现自己的 Bridge。", "S02"],
  ["AppWorld", "EnvHarness / Agent", "EnvHarness; AppWorld", "多应用 API", "跨消息、购物、日历等 API 完成状态化任务", "状态单测，含 collateral damage", "可控世界与任务状态", "状态、API 契约、故障、跨应用链", 2,2,2,2,1,2, "外部 benchmark 中最接近项目目标；验证器强，适合 Contract 故障。", "S03"],
  ["τ-bench", "Agent", "τ-bench", "数据库 + API + 模拟用户", "按政策改签航班或处理零售订单", "终态数据库 + 注释目标", "任务 DB 可重置", "工具契约、用户条件、返回错误", 2,2,2,2,1,2, "重复调用、政策约束和状态恢复都与项目高度一致。", "S04"],
  ["ALFWorld", "EnvHarness / Env", "EnvHarness; ALFWorld", "文本具身", "清洗物品后放入容器", "环境目标谓词", "确定性任务实例", "Stage 起点、动作限制、观察屏蔽", 2,2,2,2,1,0, "成本较低且是 EnvHarness 主实验；但与 API 副作用问题距离较远。", "S05"],
  ["WebShop", "EnvHarness / Agent", "EnvHarness RL; WebShop", "模拟电商网页", "按属性搜索并购买商品", "属性匹配得分/成功", "可重置站点", "观察、导航、预算、故障", 2,2,2,2,1,0, "适合测试网页环境包装，部署比真实网站稳定。", "S06"],
  ["WebArena", "EnvHarness / Agent", "EnvHarness; WebArena", "多站点浏览器", "在论坛、购物、GitLab 等完成长程任务", "任务级执行 evaluator", "Docker 网站可重置", "Stage 页面起点、Contract 导航/观察", 1,2,2,2,0,1, "跨域价值高，但运行与状态恢复成本显著。", "S07"],
  ["WorkArena / BrowserGym", "Agent harness", "WorkArena; BrowserGym", "企业 Web UI", "ServiceNow 知识工作流程", "任务级程序化 evaluator", "远端/本地实例可重置", "浏览器动作、观察、任务参数", 1,2,1,2,0,1, "BrowserGym 是潜在统一 Bridge；ServiceNow 部署较重。", "S08"],
  ["SWE-bench Verified", "EnvHarness / Agent RSI", "EnvHarness; DGM; SICA", "容器化代码仓库", "修复真实 GitHub issue 并通过测试", "Fail-to-Pass / Pass-to-Pass 测试", "每任务容器重建", "仓库起点、shell 契约、超时、预算", 1,2,2,2,0,1, "验证强、相关工作多；但单轮成本高，不宜作为最初环境搜索内环。", "S09"],
  ["OfficeQA", "EnvHarness / Agent RSI", "EnvHarness; MetaSkill-Evolve", "文档检索与推理", "从密集财务文档找数并组合答案", "EM/F1", "静态语料", "检索观察、上下文预算", 1,1,2,1,2,0, "容易运行但可修改的环境状态较少，更适合观察契约研究。", "S10"],
  ["SpreadsheetBench", "EnvHarness / Agent", "EnvHarness; SpreadsheetBench", "电子表格", "修改复杂工作簿并保留结构", "工作簿结果评分", "文件副本重置", "初始文件、工具契约、故障、预算", 1,2,2,2,0,1, "适合文件副作用和工具错误，但 Excel 运行时增加平台成本。", "S11"],
  ["AgentBench", "General Agent", "AgentBench", "八类交互环境", "OS、数据库、网页、游戏等多轮任务", "环境原生指标", "各子环境不同", "依赖子环境", 0,1,1,1,1,1, "覆盖面好，统一修改与快照能力弱；适合作为后期泛化评测。", "S12"],
  ["AgentBoard", "General Agent", "AgentBoard", "多 benchmark 评测层", "分析多轮任务的阶段进度", "终局 + progress rate", "依赖底层环境", "主要做评测，不直接改环境", 0,1,1,0,2,1, "适合借鉴过程指标，不是环境演化首选底座。", "S13"],
  ["OSWorld", "General Agent", "OSWorld", "真实桌面 OS", "跨浏览器、Office、文件和多应用工作流", "134 类执行式 evaluator", "VM/镜像恢复", "初始状态、GUI 观察、系统故障", 1,2,1,2,0,0, "高度真实但成本和非确定性都高；MVP 后再接入。", "S14"],
  ["AndroidWorld", "General Agent / Env", "AndroidWorld", "Android 模拟器", "20 个应用中的参数化移动任务", "setup/success/teardown 程序", "模拟器重置", "任务参数、设备状态、观察", 2,2,1,2,0,0, "动态任务设计值得借鉴；设备运行时不适合首版。", "S15"],
  ["Terminal-Bench 2.0", "Agent harness", "Terminal-Bench", "容器化终端", "真实系统、数据与软件操作任务", "综合测试", "每任务独立环境", "文件、命令、资源限制、故障", 1,2,2,1,0,1, "很适合验证 harness 差异；任务异构使自动环境变换较难。", "S16"],
  ["Pokémon Red / Emerald", "Agent harness RSI", "Continual Harness", "长程游戏模拟器", "不重置地持续推进游戏里程碑", "里程碑/进度", "刻意不重置", "记忆与 harness 为主，环境改动少", 0,1,0,1,0,0, "说明长程持续适应，但与可重置 EnvHarness MVP 冲突。", "S17"],
  ["Factorio", "Agent harness", "Prime Agent", "开放式游戏", "采集、建造与技术推进", "进度与任务指标", "世界存档", "世界状态、资源、多人协作", 1,1,1,1,0,0, "长程性强，但难做便宜、严格的环境候选验证。", "S18"],
  ["Polyglot", "Agent RSI", "Darwin Gödel Machine", "多语言代码修复", "跨语言仓库问题修复", "测试", "任务隔离", "主要用于评价 Agent 代码进化", 0,2,1,0,1,0, "是 Agent RSI 对照，不是环境变换底座。", "S19"],
  ["LiveCodeBench", "Agent RSI", "SICA", "持续更新代码题", "生成通过隐藏测试的程序", "隐藏测试", "无状态任务", "题面/测试，交互环境很弱", 0,2,2,0,2,0, "污染控制好，但几乎没有可交互环境状态。", "S33"],
  ["SkillsBench", "Agent skill RSI", "SkillsBench; EvoSkills", "11 个专业领域", "有/无 skill 完成文件或工具任务", "确定性 verifier", "任务隔离", "环境变化不是主变量", 1,2,1,0,1,1, "适合后期衡量由环境轨迹生成的 skill，不适合第一阶段搜索。", "S22"],
  ["PAIRED testbeds", "Env evolution", "PAIRED", "迷宫 + MuJoCo", "生成困难但可解的导航/控制环境", "环境回报 + regret", "程序化重置", "自由参数由教师生成", 2,1,2,2,1,1, "提供可解性与能力前沿思想；迁移到 LLM 环境需强 verifier。", "S23"],
  ["ACCEL testbeds", "Env evolution", "ACCEL", "MiniGrid + BipedalWalker", "从已有 level 增量编辑课程", "regret / 回报", "程序化重置", "局部 level 编辑", 2,1,2,2,1,1, "最接近建议的“从已验证父节点局部变异”策略。", "S24"],
  ["Procgen", "Env evolution", "Procgen; PLR", "16 个程序化游戏", "训练/测试于不同关卡分布", "游戏回报", "seed 重置", "程序化 level 参数与重采样", 2,1,2,2,2,0, "廉价研究课程与重放，但任务语义离工具 Agent 较远。", "S25"],
  ["POET BipedalWalker", "Env evolution", "Enhanced POET", "连续控制地形", "共同进化地形与对应策略", "回报 + minimal criterion", "程序化重置", "地形编码、novelty、策略迁移", 2,1,2,2,0,0, "谱系与档案思想有用；原方法计算开销过高。", "S26"],
  ["OMNI-EPIC PyBullet worlds", "Env evolution", "OMNI-EPIC", "生成物理环境", "基础模型生成下一可学且有趣的环境和奖励", "生成奖励 + Agent 表现", "仿真重置", "任意环境/奖励代码", 1,0,2,2,0,0, "搜索空间最开放，也最需要 verifier 与安全沙箱；不适合 MVP。", "S27"],
];

const interfaceByDomain = {
  "状态化 API": "JSON tool calls",
  "内存 toy env": "reset / step",
  "多应用 API": "Python / JSON APIs",
  "数据库 + API + 模拟用户": "JSON tools + dialogue",
  "文本具身": "text actions / observations",
  "模拟电商网页": "browser actions",
  "多站点浏览器": "Playwright / browser actions",
  "企业 Web UI": "BrowserGym actions",
  "容器化代码仓库": "shell + file editing",
  "文档检索与推理": "search / retrieval tools",
  "电子表格": "file / spreadsheet tools",
  "八类交互环境": "environment-specific actions",
  "多 benchmark 评测层": "evaluation adapters",
  "真实桌面 OS": "GUI / accessibility actions",
  "Android 模拟器": "touch / text / app actions",
  "容器化终端": "shell actions",
  "长程游戏模拟器": "controller actions",
  "开放式游戏": "game API / code actions",
  "多语言代码修复": "shell + file editing",
  "持续更新代码题": "code submission",
  "11 个专业领域": "task-specific tools/files",
  "迷宫 + MuJoCo": "discrete / continuous control",
  "MiniGrid + BipedalWalker": "level editor + control",
  "16 个程序化游戏": "discrete control",
  "连续控制地形": "continuous control",
  "生成物理环境": "generated Python simulator",
};

const methods = [
  ["Stage / Setup", "重置后重放合法动作以改变初始状态", "轨迹诊断或手工目标", "保留原生 verifier", "起点可达、可回放、域无关", "依赖确定性 reset；动作序列可能很长", "P0：首版实现", "S01"],
  ["Contract / Rules", "过滤/改写动作、观察与转移", "轨迹诊断；能力弱点或指标目标", "保留原生终局 verifier", "最适合工具错误、部分可见与预算", "规则可能制造不可解或虚假故障", "P0：首版核心", "S01"],
  ["Chain / Link", "串联或组合多个环境", "长程能力需求", "串行时合取子 verifier", "扩大任务 horizon", "共享状态、分支与语义兼容难验证", "P2：延后", "S01"],
  ["Domain randomization / Procgen", "在预定义参数空间随机采样", "均匀或指定分布", "原环境回报", "吞吐高、边界清晰", "不针对当前 Agent，结构受人工编码限制", "作为随机基线", "S25"],
  ["PLR", "从已有 level 中优先重放", "学习潜力/TD error", "原环境回报", "无需生成新代码，成本低", "只能利用已有 level 池", "P0 控制组", "S25"],
  ["PAIRED", "教师从参数空间生成环境", "protagonist-antagonist regret", "回报与可解性代理", "产生困难但可解课程", "三方训练不稳定；regret 是近似", "借鉴能力前沿", "S23"],
  ["ACCEL", "局部编辑已存在 level", "regret", "原环境回报", "从简单到复杂、保留父子谱系", "依赖可编辑 level 编码", "最接近本项目搜索策略", "S24"],
  ["POET", "环境与配对策略共同进化并跨环境迁移", "novelty + minimal criterion", "最终可解决条件", "保留多条 stepping-stone 路径", "计算极重且通常域特定", "只借鉴档案/DAG", "S26"],
  ["EnvGen", "LLM 生成和调整环境配置", "Agent 表现与课程目标", "环境原生或生成反馈", "配置层可扩展", "强依赖具身 simulator 配置", "P1 参考", "S28"],
  ["GenEnv", "LLM 模拟转移、观察和成功信号", "α-curriculum 难度", "生成式模拟器信号", "数据效率高、可快速扩展", "幻觉和 verifier 漂移风险", "对照，不作为 MVP", "S29"],
  ["Agent-World", "合成工具、数据库、环境和任务", "能力缺口与多环境训练", "程序化可验证任务", "可从零扩展新世界", "工程面大，生成逻辑也需验证", "长期方向", "S30"],
  ["VeriEnv", "克隆可执行网站并暴露后端状态", "站点覆盖与训练需求", "确定性后端 verifier", "网页真实性与可靠验证兼顾", "网站生成/部署成本高", "Web 方向优先参考", "S31"],
  ["OMNI-EPIC", "生成环境代码与奖励代码", "可学性 + interestingness", "生成奖励和 Agent 表现", "开放式、表达能力强", "奖励投机、代码安全、可解性验证困难", "远期开放式探索", "S27"],
];

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Benchmark landscape");
const methodSheet = workbook.worksheets.add("Methods");
const sourceSheet = workbook.worksheets.add("Sources");

summary.showGridLines = false;
methodSheet.showGridLines = false;
sourceSheet.showGridLines = false;
summary.tabColor = "#1F4E78";
methodSheet.tabColor = "#5B9BD5";
sourceSheet.tabColor = "#A5A5A5";

summary.getRange("A2:S2").format.font = { name: font, size: 14, bold: true, color: "#1F1F1F" };
summary.getRange("A2").values = [["Agent / Harness / RSI / Environment benchmark landscape"]];
summary.getRange("A3:S3").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };
summary.getRange("A4:F4").values = [["P0 count", null, "P1 count", null, "P2 count", null]];
summary.getRange("A5:F5").formulas = [["=COUNTIFS(A8:A33,\"P0\")", null, "=COUNTIFS(A8:A33,\"P1\")", null, "=COUNTIFS(A8:A33,\"P2\")", null]];
summary.getRange("A4:F4").format = { font: { name: font, size: 10, bold: true, color: "#44546A" }, verticalAlignment: "center" };
summary.getRange("A5:F5").format = { font: { name: font, size: 12, bold: true, color: "#1F4E78" }, verticalAlignment: "center" };
summary.getRange("H4:S5").values = [["评分规则", "六项各 0–2 分：状态可控、验证器、重置、变换适配、低搭建成本、项目贴合。10–12=P0，7–9=P1，0–6=P2。", null,null,null,null,null,null,null,null,null,null],["解释", "优先级是本项目启动顺序，不代表 benchmark 的学术质量。", null,null,null,null,null,null,null,null,null,null]];
summary.getRange("H4:H5").format.font = { name: font, size: 10, bold: true, color: "#44546A" };
summary.getRange("I4:S5").format = { font: { name: font, size: 10, italic: true, color: "#595959" }, wrapText: false };

const summaryHeaders = ["Priority","Score","Benchmark / testbed","Family","Representative work","Domain","Interface","Example task","Verifier","Reset / snapshot","Mutation lever","Control","Verifier","Reset","Mutation fit","Low setup","Project fit","Priority reason","Source ID"];
summary.getRange("A7:S7").values = [summaryHeaders];
summary.getRange("A7:S7").format = { fill: "#1F4E78", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { insideVertical: { style: "thin", color: "#FFFFFF" }, bottom: { style: "medium", color: "#17365D" } } };

const startRow = 8;
const endRow = startRow + benchmarks.length - 1;
const valueRows = benchmarks.map((r) => [null, null, ...r.slice(0, 4), interfaceByDomain[r[3]], ...r.slice(4)]);
summary.getRange(`A${startRow}:S${endRow}`).values = valueRows;
summary.getRange(`B${startRow}:B${endRow}`).formulas = benchmarks.map((_, i) => {
  const row = startRow + i;
  return [`=SUM(L${row}:Q${row})`];
});
summary.getRange(`A${startRow}:A${endRow}`).formulas = benchmarks.map((_, i) => {
  const row = startRow + i;
  return [`=IF(B${row}>=10,"P0",IF(B${row}>=7,"P1","P2"))`];
});
summary.getRange(`A${startRow}:S${endRow}`).format.font = { name: font, size: 10, color: "#222222" };
summary.getRange(`A${startRow}:S${endRow}`).format.verticalAlignment = "top";
summary.getRange(`C${startRow}:K${endRow}`).format.wrapText = true;
summary.getRange(`R${startRow}:R${endRow}`).format.wrapText = true;
summary.getRange(`A${startRow}:B${endRow}`).format.horizontalAlignment = "center";
summary.getRange(`L${startRow}:Q${endRow}`).format.horizontalAlignment = "center";
summary.getRange(`S${startRow}:S${endRow}`).format.horizontalAlignment = "center";
summary.getRange(`A${startRow}:S${endRow}`).format.borders = { insideHorizontal: { style: "thin", color: "#E7E6E6" } };
summary.getRange(`A${startRow}:A${endRow}`).conditionalFormats.add("containsText", { text: "P0", format: { fill: "#FCE4D6", font: { bold: true, color: "#C00000" } } });
summary.getRange(`A${startRow}:A${endRow}`).conditionalFormats.add("containsText", { text: "P1", format: { fill: "#FFF2CC", font: { bold: true, color: "#9C6500" } } });
summary.getRange(`A${startRow}:A${endRow}`).conditionalFormats.add("containsText", { text: "P2", format: { fill: "#E7E6E6", font: { bold: true, color: "#595959" } } });
summary.freezePanes.freezeRows(7);
summary.freezePanes.freezeColumns(3);

const summaryWidths = [9,8,26,18,24,17,20,30,24,18,28,9,9,9,11,10,10,38,10];
summaryWidths.forEach((width, i) => summary.getRangeByIndexes(0, i, endRow, 1).format.columnWidth = width);
summary.getRange(`A${startRow}:S${endRow}`).format.rowHeight = 58;
summary.getRange("A7:S7").format.rowHeight = 34;

methodSheet.getRange("A2:H2").format.font = { name: font, size: 14, bold: true, color: "#1F1F1F" };
methodSheet.getRange("A2").values = [["Environment evolution method map"]];
methodSheet.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };
methodSheet.getRange("A5:H5").values = [["Method","What changes","Selection signal","Verifier strategy","Strength","Risk","Use in this project","Source ID"]];
methodSheet.getRange("A5:H5").format = { fill: "#5B9BD5", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, borders: { insideVertical: { style: "thin", color: "#FFFFFF" } } };
const methodEnd = 6 + methods.length - 1;
methodSheet.getRange(`A6:H${methodEnd}`).values = methods;
methodSheet.getRange(`A6:H${methodEnd}`).format = { font: { name: font, size: 10, color: "#222222" }, verticalAlignment: "top", wrapText: true, borders: { insideHorizontal: { style: "thin", color: "#E7E6E6" } } };
methodSheet.getRange(`H6:H${methodEnd}`).format.horizontalAlignment = "center";
[24,32,24,25,28,30,24,10].forEach((width, i) => methodSheet.getRangeByIndexes(0, i, methodEnd, 1).format.columnWidth = width);
methodSheet.getRange(`A6:H${methodEnd}`).format.rowHeight = 54;
methodSheet.getRange("A5:H5").format.rowHeight = 32;
methodSheet.freezePanes.freezeRows(5);

sourceSheet.getRange("A2:E2").format.font = { name: font, size: 14, bold: true, color: "#1F1F1F" };
sourceSheet.getRange("A2").values = [["Primary sources"]];
sourceSheet.getRange("A3:E3").format.borders = { bottom: { style: "thin", color: "#9EADBA" } };
sourceSheet.getRange("A5:E5").values = [["Source ID","Work","Year","Primary URL","Evidence used"]];
sourceSheet.getRange("A5:E5").format = { fill: "#44546A", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", borders: { insideVertical: { style: "thin", color: "#FFFFFF" } } };
const sourceEnd = 6 + sources.length - 1;
sourceSheet.getRange(`A6:E${sourceEnd}`).values = sources;
sourceSheet.getRange(`A6:E${sourceEnd}`).format = { font: { name: font, size: 10, color: "#222222" }, verticalAlignment: "top", wrapText: true, borders: { insideHorizontal: { style: "thin", color: "#E7E6E6" } } };
sourceSheet.getRange(`A6:A${sourceEnd}`).format.horizontalAlignment = "center";
sourceSheet.getRange(`C6:C${sourceEnd}`).format.horizontalAlignment = "center";
[10,42,9,58,46].forEach((width, i) => sourceSheet.getRangeByIndexes(0, i, sourceEnd, 1).format.columnWidth = width);
sourceSheet.getRange(`A6:E${sourceEnd}`).format.rowHeight = 42;
sourceSheet.getRange("A5:E5").format.rowHeight = 28;
sourceSheet.freezePanes.freezeRows(5);

workbook.recalculate();

const summaryCheck = await workbook.inspect({ kind: "table", range: `Benchmark landscape!A2:S${endRow}`, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 19, maxChars: 12000 });
const methodCheck = await workbook.inspect({ kind: "table", range: `Methods!A2:H${methodEnd}`, include: "values,formulas", tableMaxRows: 8, tableMaxCols: 8, maxChars: 6000 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range, file] of [
  ["Benchmark landscape", "A1:S16", "benchmark.png"],
  ["Benchmark landscape", "A17:S33", "benchmark-bottom.png"],
  ["Methods", `A1:H${methodEnd}`, "methods.png"],
  ["Sources", "A1:E18", "sources.png"],
  ["Sources", `A19:E${sourceEnd}`, "sources-bottom.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${file}`, new Uint8Array(await preview.arrayBuffer()));
}

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

console.log(summaryCheck.ndjson);
console.log(methodCheck.ndjson);
console.log(errors.ndjson);
console.log(JSON.stringify({ outputPath, previewDir, benchmarkRows: benchmarks.length, sourceRows: sources.length }));
