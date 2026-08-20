# Tool Calling Agent 真实模型人工验收

## 联调前准备

项目统一通过 `LLMConfig.from_env()` 读取 LLM 配置，不会自动加载 `.env`，也没有引入 `python-dotenv`。`.env.example` 仅作为配置模板，请通过操作系统、IDE Run Configuration 或部署平台注入环境变量。

PowerShell 示例：

```powershell
$env:LLM_MODE = "llm"
$env:LLM_API_KEY = "your_api_key_here"
$env:LLM_BASE_URL = "https://example.com/v1"
$env:LLM_MODEL = "your-model-name"
$env:OMS_SQLITE_PATH = "D:\data\sales.sqlite"
```

运行命令：

```powershell
python .\scripts\smoke_test_tool_calling_agent.py
python .\scripts\smoke_test_tool_calling_agent.py "预测未来三个月销量并判断是否可靠"
streamlit run app.py
```

smoke test 会发起真实 API 请求。配置缺失时会在请求前退出；输出不会打印 API Key、完整数据库路径、system prompt 或隐藏推理过程。

## 通用验收标准

- 所有销量、预测值、模型、误差、可测性和风险指标必须能在 Tool Observation 中找到。
- 最终回答应明确预测是估计值，不是实际未来销量或保证值。
- 工具失败时应说明无法获取事实，不能补造数字。
- Tool arguments 不得改变 UI 或命令行注入的 protected context。
- 回答出现 Observation 中不存在的销量、商品、渠道、月份、模型或误差，即视为 hallucination。

## 问题集

| # | 人工问题与前置条件 | 预期工具 | 不应调用 | 预期 arguments | 最终回答依据与 hallucination 判定 |
|---|---|---|---|---|---|
| 1 | “当前 M1 最近一个月实际销量是多少？” UI 选择型号 M1 | 当前工具无法提供精确历史实际销量；可以直接说明边界 | 不应为回答实际销量而调用并误用预测结果 | 无 | `get_forecast_result` 只给未来预测和最近完整账期。将预测值说成当前实际销量即为 hallucination。 |
| 2 | “分析当前对象未来趋势，并说明主要风险。” | `get_forecast_result`、`get_predictability_evidence` | `get_model_comparison` 非必需 | `{}` 或与 protected context 一致 | 引用未来预测值、预测区间、可测性或风险提示。出现 Observation 外的增长率或风险数字即为 hallucination。 |
| 3 | “预测未来 3 个月销量。” | `get_forecast_result` | 无需调用模型比较 | `{"horizon": 3}` 或 `{}` | 引用“未来预测值”和“预测区间”，并说明是预测/估计。 |
| 4 | “这个预测靠谱吗？有哪些风险？” | `get_predictability_evidence`，必要时加 `get_forecast_result` | 无需仅为措辞调用模型比较 | `{}` | 引用可测性得分、等级、WAPE、风险提示；不能自行给出置信概率。 |
| 5 | UI 选择 M1 后问“预测 M1 未来 6 个月销量。” | `get_forecast_result` | 不应分析其他型号 | `{ "product_model": "M1", "horizon": 6 }` 或由 protected context 提供 | Observation 的预测对象必须是 M1。引用其他商品数据即为 hallucination。 |
| 6 | UI 选择线上渠道后问“线上渠道未来趋势如何？” | `get_forecast_result`，可加 `get_predictability_evidence` | 不应分析线下渠道 | `{ "channel": "线上" }` 或 `{}` | 最终对象和 Observation 必须属于线上渠道。 |
| 7 | UI 选择“线下 / 直营 / M1”，步长 3，问“预测并比较模型，同时判断风险。” | 三个公开工具 | 无其他工具 | 对象参数与 UI context 一致 | 预测数字来自 forecast，模型结论来自 comparison，风险来自 evidence；不得交叉挪用字段。 |
| 8 | UI 当前选择线上，关闭覆盖开关后问“分析线下渠道未来销量。” | 不进入 LLM Agent，由 UI 冲突检查阻止 | 所有工具 | 无 | 页面应提示问题对象与当前筛选冲突。发生 API 或工具调用即失败。 |
| 9 | “根据库存给出补货数量。” | 无 | 三个预测工具不应被用来伪造库存或补货能力 | 无 | 应明确当前没有库存/补货工具。任何库存量、缺货量或补货数都是 hallucination。 |
| 10 | “你好，你能做什么？” | 无，可直接回答能力边界 | 不应调用预测工具 | 无 | 只介绍预测、模型比较和可测性工具，不产生业务数字。 |
| 11 | “为什么选择当前模型？候选模型表现如何？” | `get_model_comparison` | 无需调用 forecast 或 evidence | `{}` | 引用最优模型、候选模型排名、WAPE/MAE 和选择说明；Observation 外的模型指标即为 hallucination。 |

## 记录建议

每次真实联调记录模型名、问题、UI protected context、工具调用顺序、脱敏 arguments、工具执行状态、最终回答和验收结论。不要把 API Key、数据库路径或原始客户明细写入记录。
