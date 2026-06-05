# MSA 合成训练数据介绍（PPT 讲稿版）

> 目的：基于当前 `msa_pretrain_64docs.jsonl` 的校验与 token 统计结果，系统说明：
> 1. MSA 对训练数据有哪些要求；
> 2. 我们当前合成训练数据有哪些特征；
> 3. 当前数据符合了哪些要求；
> 4. 哪些要求仍不一定满足，需要后续补充验证。

---

## Slide 1 — 标题

# MSA 合成训练数据质量与适配性分析

- 数据文件：`converted_training_data/msa_pretrain_64docs.jsonl`
- 数据构造目标：64 documents per sample，1 positive + 63 negatives
- 校验结果：17,132,210 条样本，1 条 JSON 解析失败
- token 统计结果：约 **564.24B tokens**

---

## Slide 2 — 为什么 MSA 训练数据要求不同于普通 LM 数据

MSA 不是只做普通 next-token prediction。它需要训练模型同时具备：

1. **长上下文生成能力**
   - 仍需要 `query/answer` 或连续文本上的 LM/answer loss。

2. **文档级 memory routing 能力**
   - 模型需要从多个 memory documents 中选出与 query 相关的文档。

3. **document-wise position reset**
   - 每个 document segment 内位置从 0 重新计数，避免长上下文位置漂移。

4. **token-level doc identity**
   - 训练 batch 里不能只有 `input_ids`，还需要 `doc_ids` 来区分 template、query/answer、document tokens。

---

## Slide 3 — MSA 代码里的硬性 token 区域语义

MSA attention 代码依赖三类 `doc_ids`：

| token 区域 | `doc_ids` 值 | 语义 |
|---|---:|---|
| Template prefix | `-2` | 模板/系统前缀 |
| Query / Answer | `0` | routing query 区域 + 生成/答案区域 |
| Memory documents | `> 0` | 文档 token，正整数是文档身份 |

这不是文档建议，而是模型 forward 中实际使用的 mask 条件。

---

## Slide 4 — MSA 最小 JSONL 样本 schema

最小训练样本应包含：

```json
{
  "sample_id": "string",
  "documents": [
    {"doc_id": 1, "text": "..."}
  ],
  "query": "string",
  "answer": "string",
  "relevant_doc_ids": [1],
  "task_type": "qa"
}
```

关键点：

- `documents` 必须非空；
- 每个 document 必须有正整数 `doc_id` 和非空 `text`；
- `query` / `answer` 必须非空；
- `relevant_doc_ids` 推荐提供，用于 routing auxiliary loss。

---

## Slide 5 — MSA 对文档与路由监督的要求

对 routing 训练来说，数据不仅要有 documents，还要能构造：

1. **正样本文档**
   - `relevant_doc_ids` 中的文档。

2. **负样本文档**
   - `hard_negative_doc_ids` 或非 relevant 的 documents。

3. **batch_aux_labels**
   - 训练时由 `relevant_doc_ids` 映射到多 hot / binary label。

4. **每个 positive id 必须存在于 documents 中**
   - 否则 aux loss 标签与候选文档集合不对齐。

---

## Slide 6 — MSA 对 position_ids 的要求

MSA 的文档位置设计要求：

- 每个 document segment 的 `position_ids` 从 0 开始；
- query/answer 区域使用自己的连续位置；
- 不能把多个 document 当成一个普通长文本连续编号；
- 如果 packing 多样本，必须避免跨样本破坏 document 边界。

当前建议：

```json
"packing": false
```

原因：MSA 的 `doc_ids` / `position_ids` 语义更复杂，普通 packing 容易破坏文档边界。

---

## Slide 7 — 当前合成数据的 manifest 关键信息

当前数据构造策略：

- mode：`expanded_same_source_negatives`
- target documents per sample：64
- max positives：1
- max negatives：16（加 supplemental negatives 后达到 64 docs）
- require target documents：true
- supplemental negatives：true
- samples written：18,504,567（manifest 原始写入量）

这说明构造目标明显偏向 **MSA routing / retrieval-style pretraining**。

---

## Slide 8 — JSONL schema 校验结果

`validate_msa_jsonl.py` 全量校验结果：

| 指标 | 数值 |
|---|---:|
| scanned records | 17,132,210 |
| valid records | 17,132,209 |
| invalid records | 1 |
| valid ratio | 0.9999999416 |
| top error | 1 条 `json_parse_error` |

解读：

- 结构质量非常高；
- 唯一坏样本是 JSON 解析错误，不是 schema mismatch；
- 可训练前过滤该行或重导该 source。

---

## Slide 9 — 文档数与正负样本分布

校验结果显示：

| 分布项 | 结果 |
|---|---:|
| documents per sample | 64 for 17,132,209 samples |
| relevant docs per sample | 1 for 17,132,209 samples |
| hard negatives per sample | 63 for 17,132,209 samples |

说明：

- 数据高度规则化；
- 每条样本几乎都是 **1 positive + 63 negatives**；
- 非常适合训练 document router / sparse retrieval 目标。

---

## Slide 10 — token 统计结果总览

`count_msa_jsonl_tokens.py` 全量 token 统计：

| 指标 | 数值 |
|---|---:|
| records processed | 17,132,210 |
| bad records | 1 |
| total tokens | 564,239,448,036 |
| total tokens (B) | 564.24B |
| avg tokens / record | 32,934.42 |

解读：

- 当前语料 token 总量非常大；
- 平均每条样本约 33k tokens；
- 说明很多样本已经接近长上下文训练规模。

---

## Slide 11 — token 来源分布

| token 来源 | token 数 | 占比 |
|---|---:|---:|
| documents | 563,346,718,106 | 99.84% |
| query | 807,068,885 | 0.14% |
| answer | 85,661,045 | 0.015% |
| total | 564,239,448,036 | 100% |

核心结论：

- token 几乎全部来自 documents；
- query/answer 占比极低；
- 这非常像 memory-heavy/router-heavy 训练数据，而不是普通 QA SFT 数据。

---

## Slide 12 — 与官方 README 中 158.95B 的差异

README 提到：

- MSA continuous pretraining：158.95B tokens
- 后续两阶段 SFT：8k → 64k curriculum

当前数据：

- 564.24B tokens
- 约为 158.95B 的 **3.55 倍**

解读：

- 当前语料规模显著大于 README 描述的 continuous pretraining token 数；
- 可能原因：64docs 展开、supplemental negatives、documents 未按目标长度截断、统计口径不同。

---

## Slide 13 — 当前数据符合了哪些 MSA 要求

### 已符合 / 强正向信号

1. **多文档结构**
   - 每条有效样本 64 docs。

2. **正负文档候选结构**
   - 1 positive + 63 hard negatives。

3. **routing supervision 可构造**
   - 每条有 1 个 relevant doc。

4. **schema 稳定性高**
   - 17M+ 样本中仅 1 条 JSON parse error。

5. **长上下文 token 规模足够**
   - 平均每条约 33k tokens，总量 564B。

---

## Slide 14 — 当前数据不一定满足的 MSA 要求

### 仍需验证 / 不一定满足

1. **token-level `doc_ids` 是否正确**
   - JSONL 有 `documents` 不等于 collator 输出一定有正确 `doc_ids`。

2. **document-wise position reset 是否正确**
   - 必须验证 collator 产出的 `position_ids` 是否每个 doc 从 0 开始。

3. **query/answer 是否正确放在 `doc_ids == 0` 区域**
   - 否则 router query mask 可能错误。

4. **template prefix 是否存在 `doc_ids == -2`**
   - 对 prefill/inference-style pipeline 重要。

5. **LM labels 是否有效**
   - 需要检查非 `-100` label 比例，否则 LM loss 可能失真。

---

## Slide 15 — 从训练日志看出的潜在风险

你提供的训练日志显示：

- `router_recall@1/@5/@10` 很快达到 1.0；
- `lm_loss` 很快接近 0；
- `aux_loss` 约等于总 loss；
- `answer_loss` 和 `reconstruction_loss` 为 0。

风险解释：

- 当前训练可能主要在优化 router auxiliary objective；
- LM 目标可能被 mask、过易或存在答案泄漏；
- routing 任务可能过于简单，或 positive/negative 区分有 source/format 泄漏。

---

## Slide 16 — 数据规模风险：token 远超官方描述

当前数据 token 总量：564.24B。

如果目标接近 README 的 158.95B，需要缩减：

```text
158.95 / 564.24 ≈ 0.282
```

也就是：

- 保留约 28.2% 样本；或
- 把 avg tokens/record 从 32,934 降到约 9,278；或
- 减少 documents 数 / document 截断长度 / negative 数。

---

## Slide 17 — 推荐的后续验证 checklist

### 必做验证

1. 抽样训练 batch，检查：
   - `doc_ids == -2` 是否存在；
   - `doc_ids == 0` 是否存在；
   - `doc_ids > 0` 是否存在；
   - 每条 sample 正 doc 数是否合理。

2. 检查 position reset：
   - 每个 document segment 的 `position_ids` 是否从 0 开始。

3. 检查 labels：
   - 非 `-100` label token 占比；
   - answer 区是否有监督；
   - document 区是否按预期参与/不参与 LM loss。

4. 检查 routing 难度：
   - positive/negative 是否存在格式泄漏；
   - 是否同 source negative 太容易；
   - 是否需要 cross-source / semantic hard negatives。

---

## Slide 18 — 推荐配置方向：如果目标是 64k 上下文

如果坚持 64docs + 64k 上下文，可以考虑：

```json
"sequence": {
  "max_seq_len": 65536,
  "max_documents_per_sample": 64,
  "max_document_tokens": 980,
  "max_query_tokens": 512,
  "max_answer_tokens": 128,
  "packing": false
}
```

说明：

- 64 × 980 + 512 + 128 ≈ 63.4k；
- 留出模板和 special tokens 空间；
- 符合长上下文训练目标，但总 token 会非常大。

---

## Slide 19 — 推荐配置方向：如果目标是接近 158.95B

如果目标接近 README 的 158.95B，且保持 17.13M 样本量：

目标平均 token / record：

```text
158.95B / 17.132M ≈ 9,278 tokens/sample
```

可考虑：

```json
"sequence": {
  "max_seq_len": 16384,
  "max_documents_per_sample": 64,
  "max_document_tokens": 140,
  "max_query_tokens": 256,
  "max_answer_tokens": 64,
  "packing": false
}
```

说明：

- 64 × 140 + 256 + 64 ≈ 9,280；
- token 预算更接近官方 continuous pretraining 规模；
- 仍保留 64docs routing 结构。

---

## Slide 20 — 总结

### 当前数据的优点

- schema 质量极高；
- 64docs 结构稳定；
- 1 positive + 63 negatives 适合 router 训练；
- token 规模足够覆盖长上下文训练。

### 当前数据的主要风险

- token 总量远超 README 中 158.95B；
- documents 占比 99.84%，LM/answer token 占比极低；
- batch 级 `doc_ids` / `position_ids` / labels 仍需验证；
- routing 任务可能过易，训练日志中 router recall 很快饱和。

### 下一步建议

1. 先做 batch-level collator 验证；
2. 检查 label 有效比例；
3. 决定目标：64k 长上下文优先，还是 158.95B token budget 优先；
4. 根据目标调整 `max_document_tokens`、样本采样比例或 negatives 策略。
