# Product Knowledge

此目录用于商品资料知识的说明性文件。结构化商品记录位于 `data/product_knowledge/`，由 `ProductKnowledgeRetriever` 读取。

商品资料只保存有来源证据的品牌、名称、分类、规格、功能和描述。未知字段保持为空，不根据型号猜测商品属性。当前销量、预测值和可测性结果不属于 Product Knowledge，必须由 SQLite 业务工具提供。
