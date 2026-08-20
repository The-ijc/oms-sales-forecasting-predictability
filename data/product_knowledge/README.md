# Product Knowledge Data Contract

Retriever 支持 UTF-8 JSON 和 JSONL。JSON 文件可以是记录数组、单条记录，或包含 `products` 数组的对象。

每条记录字段：

- `model`: 原始商品型号。
- `normalized_model`: 仅移除型号分隔符并转为大写后的稳定匹配值。
- `brand`, `category`, `subcategory`, `product_name`, `launch_date`: 字符串，未知时为空。
- `specifications`: JSON object，未知时为空对象。
- `features`: 字符串数组，未知时为空数组。
- `description`: 字符串，未知时为空。
- `sources`: 来源数组，每项只保留 `title`, `url`, `source_type`, `retrieved_at`。

采集 Pipeline 不得在记录中保存网页本地绝对路径，也不得在没有可靠来源时推断商品信息。
