# V2.2 受控 LLM 增强解释层

## 1. 架构图

```text
Streamlit 智能分析助手
        |
        v
问题对象与 sidebar context 一致性校验
        |
        v
V2.1 规则驱动 Agent
        |
        v
本地预测工具 / 模型比较 / 可测性证据
        |
        v
受控 fact pack
        |
        v
LLM 增强解释（可选）
        |
        v
事实一致性校验
        |
        +--> 通过：展示 LLM 业务解读
        |
        +--> 失败：回退规则驱动回答
```

LLM 不直接访问 SQLite，不直接读取客户明细，不直接生成预测值。

## 2. 数据流

1. 页面读取当前 sidebar 的预测层级、渠道大类、渠道细分类、型号、预测步长、回测窗口和数据库路径。
2. 页面先检查用户问题中的对象关键词是否与当前筛选一致。
3. 一致或用户明确允许覆盖后，调用 V2.1 `analyze_sales_question(question, context)`。
4. V2.1 调用本地工具生成预测值、模型比较、可测性证据和风险提示。
5. V2.2 从工具结果构建 fact pack。
6. 仅当 `LLM_MODE=llm` 且配置完整时，调用 OpenAI-compatible Chat Completions API。
7. LLM 返回 JSON 后必须通过事实一致性校验，未通过则不展示。

## 3. fact pack 字段

fact pack 只包含允许 LLM 引用的聚合事实：

- `actual_analysis_object`：实际分析对象；
- `forecast_horizon`：预测周期；
- `best_model`：本地工具选择的最优模型；
- `allowed_models`：工具结果中出现过的模型名称；
- `forecast_values`：工具计算出的未来预测值；
- `forecast_interval`：工具计算出的经验预测区间；
- `recent_complete_period`：最新完整账期；
- `model_metrics`：WAPE、sMAPE、MAE、Bias 等工具指标；
- `predictability`：可测性得分、等级和证据；
- `risk_warnings`：本地工具生成的风险提示；
- `tool_calls`：本次调用的工具名称；
- `business_scope`：固定业务边界说明。

## 4. LLM 输出 JSON 格式

LLM 必须只返回合法 JSON：

```json
{
  "executive_summary": "不超过 180 字，不允许编造数值",
  "business_interpretation": "业务解读",
  "risk_recommendations": ["建议1", "建议2"],
  "follow_up_questions": ["建议用户继续查看的问题"],
  "evidence_references": ["引用的事实字段名称"]
}
```

## 5. 事实一致性校验

系统会做保守校验：

- 返回必须是 JSON 对象；
- 必须包含规定字段；
- 文本长度不能超过限制；
- 输出中出现的模型名必须来自 fact pack；
- 输出中出现的月份、数值、百分比不能与 fact pack 明显矛盾；
- 不得出现“保证准确率”；
- 不得建议上传客户明细、数据库文件或原始交易数据。

校验失败时，页面显示规则驱动分析，并提示 LLM 输出未通过事实一致性校验。

## 6. 降级策略

以下情况自动降级到 V2.1：

- `LLM_MODE=rule`；
- 缺少 `LLM_API_KEY`、`LLM_BASE_URL` 或 `LLM_MODEL`；
- API 调用失败、超时或返回结构异常；
- LLM 输出不是合法 JSON；
- LLM 输出缺字段；
- LLM 输出包含 fact pack 外的模型、月份或明显不一致数值。

## 7. API Key 安全

- 不要把真实 API Key 写入代码、README、测试或 `.env.example`。
- `.env` 和 `.env.*` 已在 `.gitignore` 中忽略。
- 页面和错误状态会避免展示 API Key。
- 测试使用 mock client，不发起真实网络请求。

## 8. 未来接入 RAG 的位置

未来可在 fact pack 构建后、LLM 调用前加入 RAG 检索层，只允许检索非敏感知识文档，例如模型选择规则、指标解释和可测性评分规则。RAG 返回内容也应进入受控 fact pack 或单独的引用包，并经过相同安全边界。

## 9. 当前限制

- 数值一致性校验是保守的基础校验，不做复杂自然语言数学推理。
- LLM 仅做解释与表达，不能改变本地工具计算结果。
- 当前不支持让 LLM 直接查询数据库或读取原始业务明细。
- 若 LLM 输出可疑，系统会宁可回退，也不展示。
