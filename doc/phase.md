# litehybrid 模块拆解与 Phase 1 实现计划

> 基于 `litehybrid.md` 的产品定位，按"深度模块 + 信息隐藏"原则拆解实现模块。

---

## 1. 设计原则

1. **按知识归属拆分，不按时间顺序拆分**
   - 不要按 `insert / search / delete` 拆成独立模块
   - 要按 `storage / index / fusion / planner` 等职责拆分
2. **每个模块隐藏一个重要设计决策**
   - `storage` 隐藏 BLOB 序列化格式
   - `planner` 隐藏多路召回执行策略
   - `fusion` 隐藏排序数学
3. **对外接口简单，内部实现可以复杂**
   - `HybridIndex::search(query)` 是主要 facade
   - 调用者不需要知道内部有几个索引
4. **免费版和商业版在编译期切分**
   - 通过 Cargo feature flag 控制
   - 避免运行时大量 `if pro` 分支污染代码

---

## 2. 顶层 Crate 结构

```
litehybrid/
├── crates/
│   ├── litehybrid-core/        # 免费版核心引擎
│   ├── litehybrid-pro/         # 商业版扩展（依赖 core）
│   ├── litehybrid-ext/         # SQLite 扩展入口（C ABI 封装）
│   └── litehybrid-cli/         # 本地调试/测试工具（Phase 1 后期可选）
```

| Crate | 职责 | 备注 |
|---|---|---|
| `litehybrid-core` | 混合检索引擎核心 | 不依赖 SQLite C API，可独立测试 |
| `litehybrid-pro` | 商业版索引和融合策略 | 通过 feature flag 编译 |
| `litehybrid-ext` | SQLite virtual table 协议适配 | 只负责把 core API 映射到 vtab |
| `litehybrid-cli` | 命令行调试工具 | 可选，Phase 1 后期再加 |

---

## 3. `litehybrid-core` 内部模块

### 3.1 `storage` —— 存储层

- **隐藏的信息**：SQLite 表结构、BLOB 序列化、rowid 映射、版本元数据
- **对外接口**：
  ```rust
  impl Segment {
      fn insert(&mut self, rowid: i64, doc: &Document) -> Result<()>;
      fn delete(&mut self, rowid: i64) -> Result<()>;
      fn get(&self, rowid: i64) -> Result<Option<Document>>;
      fn iter(&self) -> impl Iterator<Item = Document>;
  }
  ```
- **设计要点**：所有持久化必须走这里，`index` 模块不直接操作 SQLite

### 3.2 `index` —— 向量索引抽象

- **隐藏的信息**：Flat / IVF / HNSW 具体算法、构建参数、量化细节
- **对外接口**：
  ```rust
  trait VectorIndex {
      fn add(&mut self, rowid: i64, vector: &[f32]) -> Result<()>;
      fn remove(&mut self, rowid: i64) -> Result<()>;
      fn search(&self, query: &[f32], topk: usize, filter: &ScalarFilter) -> Result<Vec<ScoredRowId>>;
      fn serialize(&self) -> Result<Vec<u8>>;
      fn deserialize(data: &[u8]) -> Result<Self> where Self: Sized;
  }
  ```
- **实现**：
  - 免费版：`FlatIndex`
  - 商业版：`IvfIndex`、`HnswIndex`

### 3.3 `metrics` —— 距离函数

- **隐藏的信息**：L2 / Cosine / Dot 实现、SIMD 优化、向量归一化
- **对外接口**：
  ```rust
  enum Metric { L2, Cosine, Dot }
  impl Metric {
      fn compute(&self, a: &[f32], b: &[f32]) -> f32;
  }
  ```

### 3.4 `fts` —— 全文搜索

- **隐藏的信息**：FTS5 virtual table 创建、tokenizer 注册、BM25 rank 解析
- **对外接口**：
  ```rust
  impl FtsIndex {
      fn search(&self, text: &str, topk: usize) -> Result<Vec<ScoredRowId>>;
      fn insert(&mut self, rowid: i64, fields: &[(String, String)]) -> Result<()>;
      fn delete(&mut self, rowid: i64) -> Result<()>;
  }
  ```

### 3.5 `scalar` —— 标量过滤

- **隐藏的信息**：SQL `WHERE` 表达式解析、过滤下推策略
- **对外接口**：
  ```rust
  struct ScalarFilter { ... }
  impl ScalarFilter {
      fn parse(sql_expr: &str) -> Result<Self>;
      fn matches(&self, doc: &Document) -> bool;
      fn can_pushdown(&self) -> bool;
      fn to_sql(&self) -> String;
  }
  ```

### 3.6 `query` —— 统一查询模型

- **职责**：用单一结构体表达所有混合查询语义
- **对外接口**：
  ```rust
  struct HybridQuery {
      vector: Option<VectorQuery>,
      text: Option<TextQuery>,
      filter: Option<ScalarFilter>,
      fusion: FusionStrategy,
      topk: usize,
      trace: bool,
  }
  ```
- **价值**：`vtab` 和 `planner` 之间的通用语言，避免多入口各自为政

### 3.7 `planner` / `executor` —— 查询执行引擎

- **隐藏的信息**：多路召回协调、候选集合并、过滤下推顺序、每路 LIMIT 策略
- **对外接口**：
  ```rust
  impl HybridIndex {
      fn search(&self, query: &HybridQuery) -> Result<SearchResult>;
  }
  ```
- **内部分工**：
  - `Planner`：把 `HybridQuery` 转成执行计划
  - `Executor`：执行计划、合并候选、调用 `Fusion`、返回结果
- **注意**：这是系统最深的模块，要防止其膨胀成上帝模块

### 3.8 `fusion` —— 融合排序

- **隐藏的信息**：RRF 公式、加权求和、分数归一化
- **对外接口**：
  ```rust
  enum FusionStrategy { Rrf, WeightedSum { weights: ... } }
  impl FusionStrategy {
      fn combine(&self, candidates: Vec<RecallResult>) -> Vec<ScoredRowId>;
  }
  ```
- **商业版扩展**：`LearnedRanker`、`CrossEncoderRanker` 作为新的 strategy

### 3.9 `txn` —— 事务一致性

- **隐藏的信息**：多索引原子更新、失败回滚、删除同步
- **对外接口**：
  ```rust
  impl HybridIndex {
      fn insert(&mut self, doc: &Document) -> Result<()>;
      fn delete(&mut self, rowid: i64) -> Result<()>;
  }
  ```
- **保证**：向量索引、FTS 索引、标量存储同时成功或同时失败

### 3.10 `trace` —— 可观测性

- **隐藏的信息**：trace 格式、输出目标
- **对外接口**：
  ```rust
  struct QueryTrace { ... }
  impl HybridIndex {
      fn search_with_trace(&self, query: &HybridQuery) -> Result<(SearchResult, QueryTrace)>;
  }
  ```

---

## 4. 模块关系图

```
┌─────────────────────────────────────────┐
│            litehybrid-ext               │
│    (SQLite virtual table protocol)      │
└─────────────────┬───────────────────────┘
                  │ translate SQL MATCH
                  ▼
┌─────────────────────────────────────────┐
│         HybridIndex (facade)            │
│  ┌─────────┐ ┌─────────┐ ┌───────────┐ │
│  │ Vector  │ │   FTS   │ │  Scalar   │ │
│  │ Index   │ │ Index   │ │  Store    │ │
│  └────┬────┘ └────┬────┘ └─────┬─────┘ │
│       └─────────────┴────────────┘       │
│              Planner / Executor         │
│              Fusion / Trace             │
└─────────────────────────────────────────┘
```

---

## 5. 免费版 vs 商业版切分

### 5.1 Cargo features

```toml
# litehybrid-pro/Cargo.toml
[features]
default = []
pro = ["litehybrid-core/pro"]
enterprise = ["pro"]
```

### 5.2 功能映射

| 能力 | 免费版 | 商业版 |
|---|---|---|
| `index::FlatIndex` | ✅ | ✅ |
| `index::IvfIndex` | ❌ | ✅ |
| `index::HnswIndex` | ❌ | ✅ |
| `fusion::Rrf` / `WeightedSum` | ✅ | ✅ |
| `fusion::CrossEncoderRanker` | ❌ | ✅ |
| 基础 `trace` | ✅ | ✅ |
| 完整决策追溯 | ❌ | ✅ |

### 5.3 实现策略

- 用 trait 注册表注入商业版索引和融合策略
- `vtab` 层不需要改动
- 避免在 free 代码中写大量 `#[cfg(feature = "pro")]`

---

## 6. Phase 1 MVP 范围

### 6.1 必须实现

1. `storage` —— 单表 + BLOB 存向量
2. `metrics` —— L2 / Cosine
3. `index::FlatIndex` —— 暴力搜索
4. `fts` —— FTS5 封装
5. `scalar` —— 基础标量过滤
6. `query` —— 统一查询结构
7. `fusion` —— RRF + Weighted Sum
8. `planner/executor` —— 两路召回 + 合并
9. `txn` —— 多索引原子写入
10. `vtab` / `ext` —— SQLite 扩展入口

### 6.2 明确不做

- 自定义 graph 引擎
- Learned reranker / Cross-Encoder
- IVF / HNSW 索引
- 复杂 SQL 优化器
- 多语言 SDK

---

## 7. 主要风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| `planner` 膨胀 | 融合、下推、多路召回都塞进一个模块 | 拆成 `Planner`（生成计划）和 `Executor`（执行） |
| `storage` 与 `index` 边界模糊 | `FlatIndex` 直接读写 SQLite | 所有持久化走 `storage`，index 只操作内存结构 |
| pro 代码污染 free | 大量 `#[cfg(feature = "pro")]` | 用 trait 注册表，pro 在初始化时注入 |
| 过早优化 | Phase 1 就想做 HNSW / graph | 严格按 Phase 1 范围执行，先跑通 Flat + FTS |

---

## 8. 下一步行动

1. 在 `litehybrid-core` 中定义 `HybridQuery`、`Document`、`ScoredRowId` 等核心类型
2. 定义 `HybridIndex` facade 的接口契约
3. 先实现 `storage` + `metrics` + `FlatIndex`
4. 接入 FTS5，实现 `fts` 模块
5. 实现 `fusion` 和 `planner/executor`
6. 最后做 `litehybrid-ext` 的 SQLite vtab 封装
