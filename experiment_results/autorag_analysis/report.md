# AutoRAG results: parsed summary and selected pipeline

Root: `/home/claude/autorag/results`  
Distinct trials: 17  

Experiment families (trials sharing prompt templates and metrics): {'F1': '10, 11, 12, 13, 15, 16, 6, 7, 8, 9', 'F2': '3, 4, 5', 'F3': '2, 5 copy', 'F4': '0, 1'}. Selection is made within **F1**; other families are listed for completeness but their metrics are not comparable (different prompt templates and, most likely, a different evaluation set).

## Trials

| trial   | family   | gen_models      |   context_lengths | temperatures   | top_ps        |   n_eval_queries |   retrieval_top_k |   reranker_top_k |   n_prompts |
|:--------|:---------|:----------------|------------------:|:---------------|:--------------|-----------------:|------------------:|-----------------:|------------:|
| 0       | F4       | llama3          |                   |                |               |                3 |                20 |                3 |           3 |
| 1       | F4       | llama3          |                   |                |               |                3 |                20 |                3 |           3 |
| 10      | F1       | qwen2.5         |               512 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 11      | F1       | qwen2.5         |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 12      | F1       | qwen2.5:14b     |               512 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 13      | F1       | qwen2.5:14b     |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 15      | F1       | llama3.1        |               512 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 16      | F1       | llama3.1        |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 2       | F3       | llama3          |                   |                |               |                3 |                20 |                3 |           3 |
| 3       | F2       | llama3; mistral |                   |                |               |                3 |                20 |                3 |           3 |
| 4       | F2       | llama3; mistral |                   |                |               |                3 |                20 |                3 |           3 |
| 5 copy  | F3       | llama3          |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 5       | F2       | llama3; mistral |                   |                |               |                3 |                20 |                3 |           3 |
| 6       | F1       | deepseek-r1     |               512 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 7       | F1       | deepseek-r1     |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 8       | F1       | deepseek-r1:14b |               512 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |
| 9       | F1       | deepseek-r1:14b |              1024 | 0.1; 0.5; 1.0  | 0.1; 0.5; 1.0 |                3 |                20 |                3 |           3 |

## Retrieval (distinct metric profiles across configurations)

| module_name   |   retrieval_ndcg |   retrieval_map |   retrieval_mrr |   retrieval_f1 |   retrieval_recall |   retrieval_precision |   n_configs |
|:--------------|-----------------:|----------------:|----------------:|---------------:|-------------------:|----------------------:|------------:|
| BM25          |         0.543643 |        0.5      |        0.5      |      0.0634921 |           0.666667 |             0.0333333 |          10 |
| HybridCC      |         0.543643 |        0.5      |        0.5      |      0.0634921 |           0.666667 |             0.0333333 |           3 |
| HybridCC      |         0.666667 |        0.666667 |        0.666667 |      0.0634921 |           0.666667 |             0.0333333 |          27 |
| HybridRRF     |         0.543643 |        0.5      |        0.5      |      0.0634921 |           0.666667 |             0.0333333 |          30 |
| VectorDB      |         0.42062  |        0.333333 |        0.333333 |      0.0634921 |           0.666667 |             0.0333333 |         300 |

Selected: **HybridCC** `{'top_k': 20, 'target_modules': ('VectorDB', 'BM25'), 'weights': (0.5, 0.5), 'weight': 0.08, 'target_module_params': ({'top_k': 20, 'embedding_model': 'huggingface_all_mpnet_base_v2', 'return_metadata': 'timestamp'}, {'top_k': 20})}` from trial 6 (34.parquet); metrics {'retrieval_ndcg': 0.666667, 'retrieval_map': 0.666667, 'retrieval_mrr': 0.666667, 'retrieval_f1': 0.063492, 'retrieval_recall': 0.666667, 'retrieval_precision': 0.033333}; tied with 26 other configuration(s) of ['HybridCC'].

## Passage reranker (distinct metric profiles)

| module_name     |   passage_reranker_retrieval_f1 |   passage_reranker_retrieval_recall |   passage_reranker_retrieval_precision |   n_configs |
|:----------------|--------------------------------:|------------------------------------:|---------------------------------------:|------------:|
| ColbertReranker |                        0.333333 |                            0.666667 |                               0.222222 |          10 |
| KoReranker      |                        0.333333 |                            0.666667 |                               0.222222 |          10 |
| PassReranker    |                        0.333333 |                            0.666667 |                               0.222222 |          10 |
| RankGPT         |                        0.333333 |                            0.666667 |                               0.222222 |          10 |
| Tart            |                        0.333333 |                            0.666667 |                               0.222222 |          10 |
| Upr             |                        0.333333 |                            0.666667 |                               0.222222 |          10 |

Selected: **PassReranker** (ties broken by execution time); 60 configurations tie on all metrics: ['ColbertReranker', 'KoReranker', 'PassReranker', 'RankGPT', 'Tart', 'Upr'].

## Prompt maker

|   trial | filename   | prompt_topic   |   average_prompt_token |   prompt_maker_sem_score |   prompt_maker_meteor |   prompt_maker_rouge |   prompt_maker_bleu |
|--------:|:-----------|:---------------|-----------------------:|-------------------------:|----------------------:|---------------------:|--------------------:|
|      10 | 0.parquet  | friendly ship  |                   5195 |                 0.316318 |              0.20831  |            0.0765128 |            0.766651 |
|      10 | 1.parquet  | enemy ship     |                   5195 |                 0.316318 |              0.20831  |            0.0765128 |            0.766651 |
|      10 | 2.parquet  | your ship      |                   5201 |                 0.316318 |              0.20831  |            0.0765128 |            0.766651 |
|      11 | 0.parquet  | friendly ship  |                   5195 |                 0.335389 |              0.21254  |            0.086868  |            1.52747  |
|      11 | 1.parquet  | enemy ship     |                   5195 |                 0.335389 |              0.21254  |            0.086868  |            1.52747  |
|      11 | 2.parquet  | your ship      |                   5201 |                 0.335389 |              0.21254  |            0.086868  |            1.52747  |
|      12 | 0.parquet  | your ship      |                   5201 |                 0.484921 |              0.2653   |            0.104459  |            3.09205  |
|      12 | 1.parquet  | enemy ship     |                   5195 |                 0.484921 |              0.2653   |            0.104459  |            3.09205  |
|      12 | 2.parquet  | friendly ship  |                   5195 |                 0.484921 |              0.2653   |            0.104459  |            3.09205  |
|      13 | 0.parquet  | enemy ship     |                   5195 |                 0.359484 |              0.210798 |            0.088751  |            1.01788  |
|      13 | 1.parquet  | friendly ship  |                   5195 |                 0.359484 |              0.210798 |            0.088751  |            1.01788  |
|      13 | 2.parquet  | your ship      |                   5201 |                 0.359484 |              0.210798 |            0.088751  |            1.01788  |

Selected: prompt 1.parquet of trial 7 (3 configurations tie).

## Generator: models ranked by mean composite over their sampling grid

| model           |   context_length |   n_configs |   bleu_mean |   rouge_mean |   meteor_mean |   sem_score_mean |   bert_score_mean |   composite_mean |   ngram_mean |   semantic_mean |   composite_best |   out_tokens |   think_share |   rank_by_mean |
|:----------------|-----------------:|------------:|------------:|-------------:|--------------:|-----------------:|------------------:|-----------------:|-------------:|----------------:|-----------------:|-------------:|--------------:|---------------:|
| llama3.1        |              512 |          18 |       1.053 |        0.069 |         0.184 |            0.457 |             0.819 |            0.513 |        0.468 |           0.581 |            0.859 |      666.741 |         0     |              1 |
| deepseek-r1     |              512 |          18 |       0.756 |        0.055 |         0.163 |            0.507 |             0.829 |            0.51  |        0.267 |           0.874 |            0.591 |      993.574 |         0.653 |              2 |
| llama3.1        |             1024 |          18 |       0.847 |        0.068 |         0.181 |            0.465 |             0.819 |            0.487 |        0.414 |           0.595 |            0.676 |      663.333 |         0     |              3 |
| deepseek-r1     |             1024 |          18 |       0.819 |        0.057 |         0.165 |            0.464 |             0.827 |            0.48  |        0.299 |           0.753 |            0.621 |      882.019 |         0.644 |              4 |
| qwen2.5         |              512 |          18 |       0.664 |        0.063 |         0.168 |            0.44  |             0.818 |            0.402 |        0.318 |           0.528 |            0.502 |      804.13  |         0     |              5 |
| qwen2.5:14b     |             1024 |          18 |       0.593 |        0.061 |         0.176 |            0.384 |             0.818 |            0.363 |        0.322 |           0.424 |            0.462 |      748.63  |         0     |              6 |
| qwen2.5         |             1024 |          18 |       0.696 |        0.062 |         0.164 |            0.418 |             0.816 |            0.361 |        0.302 |           0.45  |            0.592 |      829.296 |         0     |              7 |
| deepseek-r1:14b |              512 |          18 |       0.557 |        0.05  |         0.157 |            0.439 |             0.82  |            0.347 |        0.191 |           0.582 |            0.475 |      970.389 |         0.585 |              8 |
| deepseek-r1:14b |             1024 |          18 |       0.535 |        0.049 |         0.147 |            0.381 |             0.818 |            0.254 |        0.145 |           0.418 |            0.484 |      879.741 |         0.584 |              9 |
| qwen2.5:14b     |              512 |          18 |       0.566 |        0.059 |         0.166 |            0.307 |             0.813 |            0.236 |        0.27  |           0.185 |            0.45  |      613.63  |         0     |             10 |

## Generator: top 15 configurations by composite (min-max normalized mean over metrics)

|   trial | model       |   context_length |   temperature |   top_p |   top_k |   min_p |   bleu |   rouge |   meteor |   sem_score |   bert_score |   composite_minmax |   composite_ngram |   composite_semantic |   composite_rank | is_best   |
|--------:|:------------|-----------------:|--------------:|--------:|--------:|--------:|-------:|--------:|---------:|------------:|-------------:|-------------------:|------------------:|---------------------:|-----------------:|:----------|
|      15 | llama3.1    |              512 |           0.1 |     0.5 |      10 |     0.1 |  2.103 |   0.1   |    0.227 |       0.492 |        0.822 |              0.859 |             0.953 |                0.717 |             16.2 | False     |
|      15 | llama3.1    |              512 |           0.1 |     1   |      10 |     0.1 |  2.402 |   0.078 |    0.21  |       0.506 |        0.829 |              0.834 |             0.815 |                0.862 |              7.8 | True      |
|      15 | llama3.1    |              512 |           0.1 |     1   |      90 |     0.9 |  2.094 |   0.081 |    0.214 |       0.468 |        0.822 |              0.741 |             0.797 |                0.657 |             23.8 | False     |
|      15 | llama3.1    |              512 |           0.5 |     0.1 |      90 |     0.9 |  1.481 |   0.074 |    0.205 |       0.504 |        0.823 |              0.682 |             0.632 |                0.758 |             16.8 | False     |
|      16 | llama3.1    |             1024 |           1   |     0.1 |      10 |     0.1 |  1.583 |   0.072 |    0.2   |       0.466 |        0.827 |              0.676 |             0.621 |                0.757 |             22   | False     |
|      16 | llama3.1    |             1024 |           0.1 |     1   |      10 |     0.1 |  1.714 |   0.076 |    0.191 |       0.473 |        0.821 |              0.639 |             0.632 |                0.648 |             27.2 | True      |
|      15 | llama3.1    |              512 |           0.1 |     0.1 |      10 |     0.1 |  1.193 |   0.072 |    0.212 |       0.476 |        0.822 |              0.632 |             0.598 |                0.683 |             24.2 | False     |
|      15 | llama3.1    |              512 |           0.1 |     0.1 |      90 |     0.9 |  1.038 |   0.066 |    0.205 |       0.527 |        0.823 |              0.626 |             0.517 |                0.788 |             22.2 | False     |
|       7 | deepseek-r1 |             1024 |           1   |     1   |      90 |     0.9 |  0.916 |   0.068 |    0.192 |       0.492 |        0.829 |              0.621 |             0.469 |                0.849 |             21   | False     |
|      16 | llama3.1    |             1024 |           1   |     0.1 |      90 |     0.9 |  1.1   |   0.076 |    0.191 |       0.508 |        0.821 |              0.61  |             0.535 |                0.722 |             23.4 | False     |
|      11 | qwen2.5     |             1024 |           0.1 |     1   |      90 |     0.9 |  1.186 |   0.071 |    0.183 |       0.47  |        0.826 |              0.592 |             0.493 |                0.74  |             29   | True      |
|       6 | deepseek-r1 |              512 |           0.1 |     0.5 |      90 |     0.9 |  0.939 |   0.057 |    0.178 |       0.499 |        0.834 |              0.591 |             0.36  |                0.937 |             42.2 | False     |
|      16 | llama3.1    |             1024 |           0.1 |     1   |      90 |     0.9 |  1.236 |   0.073 |    0.189 |       0.504 |        0.819 |              0.589 |             0.533 |                0.674 |             28.4 | False     |
|       6 | deepseek-r1 |              512 |           1   |     0.5 |      10 |     0.1 |  1.016 |   0.068 |    0.183 |       0.485 |        0.827 |              0.583 |             0.449 |                0.785 |             28.6 | False     |
|       6 | deepseek-r1 |              512 |           0.5 |     0.1 |      10 |     0.1 |  0.657 |   0.06  |    0.178 |       0.527 |        0.831 |              0.578 |             0.33  |                0.951 |             45.2 | False     |

## Other families (not comparable): best generator row per trial

| trial   | family   | model   | filename   |   bleu |   rouge |   meteor |   sem_score |   bert_score |
|:--------|:---------|:--------|:-----------|-------:|--------:|---------:|------------:|-------------:|
| 0       | F4       | llama3  | 0.parquet  | 48.905 |   0.693 |    0.839 |       0.852 |          nan |
| 1       | F4       | llama3  | 0.parquet  | 16.931 |   0.312 |    0.622 |       0.801 |          nan |
| 5 copy  | F3       | llama3  | 2.parquet  |  0.385 |   0.043 |    0.093 |       0.216 |          nan |
| 2       | F3       | llama3  | 0.parquet  |  0.182 |   0.034 |    0.088 |       0.245 |          nan |

## Best configuration by each single metric

|            |   trial | model       | filename   |    value |
|:-----------|--------:|:------------|:-----------|---------:|
| bleu       |      15 | llama3.1    | 2.parquet  | 2.40173  |
| rouge      |      15 | llama3.1    | 1.parquet  | 0.100363 |
| meteor     |      15 | llama3.1    | 1.parquet  | 0.226505 |
| sem_score  |       6 | deepseek-r1 | 2.parquet  | 0.532148 |
| bert_score |       6 | deepseek-r1 | 10.parquet | 0.833507 |

## AutoRAG's own per-trial best generator, re-scored on the common composite

|   trial | model           |   context_length | filename   |   bleu |   rouge |   meteor |   sem_score |   bert_score |   composite_minmax |
|--------:|:----------------|-----------------:|:-----------|-------:|--------:|---------:|------------:|-------------:|-------------------:|
|      15 | llama3.1        |              512 | 2.parquet  |  2.402 |   0.078 |    0.21  |       0.506 |        0.829 |              0.834 |
|      16 | llama3.1        |             1024 | 2.parquet  |  1.714 |   0.076 |    0.191 |       0.473 |        0.821 |              0.639 |
|      11 | qwen2.5         |             1024 | 11.parquet |  1.186 |   0.071 |    0.183 |       0.47  |        0.826 |              0.592 |
|       6 | deepseek-r1     |              512 | 11.parquet |  1.17  |   0.059 |    0.17  |       0.509 |        0.829 |              0.577 |
|       7 | deepseek-r1     |             1024 | 7.parquet  |  1.079 |   0.057 |    0.182 |       0.429 |        0.826 |              0.504 |
|      10 | qwen2.5         |              512 | 14.parquet |  1.303 |   0.063 |    0.173 |       0.478 |        0.816 |              0.491 |
|       9 | deepseek-r1:14b |             1024 | 12.parquet |  1.128 |   0.055 |    0.189 |       0.405 |        0.824 |              0.484 |
|       8 | deepseek-r1:14b |              512 | 12.parquet |  0.898 |   0.046 |    0.165 |       0.485 |        0.827 |              0.462 |
|      12 | qwen2.5:14b     |              512 | 0.parquet  |  0.988 |   0.07  |    0.209 |       0.313 |        0.819 |              0.45  |
|      13 | qwen2.5:14b     |             1024 | 9.parquet  |  1.132 |   0.063 |    0.19  |       0.39  |        0.815 |              0.431 |

## Variance decomposition of generator metrics (between model/context vs within sampling grid)

| metric     |   between_share |   within_share |   within_sd |   between_range |
|:-----------|----------------:|---------------:|------------:|----------------:|
| bleu       |           0.23  |          0.77  |       0.292 |           0.518 |
| rouge      |           0.603 |          0.397 |       0.005 |           0.021 |
| meteor     |           0.369 |          0.631 |       0.014 |           0.037 |
| sem_score  |           0.752 |          0.248 |       0.032 |           0.2   |
| bert_score |           0.651 |          0.349 |       0.004 |           0.017 |

## Paired wins between models (row beats column; summed over 5 metrics x 3 queries, medians over each model's grid; full table in model_pairwise_wins.csv)

|                         |   deepseek-r1 @1024.0 |   deepseek-r1 @512.0 |   deepseek-r1:14b @1024.0 |   deepseek-r1:14b @512.0 |   llama3.1 @1024.0 |   llama3.1 @512.0 |   qwen2.5 @1024.0 |   qwen2.5 @512.0 |   qwen2.5:14b @1024.0 |   qwen2.5:14b @512.0 |   total_wins |
|:------------------------|----------------------:|---------------------:|--------------------------:|-------------------------:|-------------------:|------------------:|------------------:|-----------------:|----------------------:|---------------------:|-------------:|
| deepseek-r1 @512.0      |                     8 |                    0 |                        14 |                       14 |                  7 |                 7 |                10 |                8 |                    10 |                   12 |           90 |
| deepseek-r1 @1024.0     |                     0 |                    7 |                        13 |                       13 |                  7 |                 8 |                10 |                9 |                    10 |                   12 |           89 |
| llama3.1 @512.0         |                     7 |                    8 |                        11 |                       12 |                  7 |                 0 |                11 |                9 |                    11 |                   10 |           86 |
| llama3.1 @1024.0        |                     8 |                    8 |                        11 |                       10 |                  0 |                 8 |                12 |                9 |                     9 |                   11 |           86 |
| qwen2.5 @512.0          |                     6 |                    7 |                        11 |                       11 |                  6 |                 6 |                12 |                0 |                     9 |                   12 |           80 |
| qwen2.5:14b @1024.0     |                     5 |                    5 |                        11 |                       10 |                  6 |                 4 |                 9 |                6 |                     0 |                    8 |           64 |
| qwen2.5 @1024.0         |                     5 |                    5 |                        12 |                       10 |                  3 |                 4 |                 0 |                3 |                     6 |                   11 |           59 |
| qwen2.5:14b @512.0      |                     3 |                    3 |                         9 |                        7 |                  4 |                 5 |                 4 |                3 |                     7 |                    0 |           45 |
| deepseek-r1:14b @512.0  |                     2 |                    1 |                         8 |                        0 |                  5 |                 3 |                 5 |                4 |                     5 |                    8 |           41 |
| deepseek-r1:14b @1024.0 |                     2 |                    1 |                         0 |                        7 |                  4 |                 4 |                 3 |                4 |                     4 |                    6 |           35 |

Head to head, deepseek-r1 @512.0 vs deepseek-r1 @1024.0, per metric (queries won by deepseek-r1 @512.0 / by deepseek-r1 @1024.0): bleu 1/2; rouge 1/2; meteor 2/1; sem_score 2/1; bert_score 2/1

## Selected generator

**{'batch_size': 1, 'llm': 'ollama', 'model': 'llama3.1', 'request_timeout': 30, 'temperature': 0.1, 'top_p': 0.5, 'top_k': 10, 'min_p': 0.1, 'max_length': 512, 'context_length': 512, 'prompt_batch_size': 16, 'repeat_penalty_tokens': 5, 'repeat_penalty': 1.2}** (trial 15, 1.parquet); metrics {'bleu': 2.102676968173791, 'rouge': 0.1003632119947183, 'meteor': 0.2265051827162778, 'sem_score': 0.4915413803900019, 'bert_score': 0.8224091728528341}; composite 0.859; AutoRAG marked it best in its own trial: False.

## Caveats the selection inherits from the experiment

* Every metric is a mean over **[3] evaluation queries**. Differences between configurations are not statistically distinguishable at this sample size; treat the ranking as a heuristic.
* Generator configurations were sampled once each at temperatures up to 1.0; a single sample per query at non-zero temperature adds noise of the same order as the between-configuration differences (see variance decomposition).
* Reasoning models emit `<think>` blocks that were scored as part of the answer; their BLEU/ROUGE/METEOR are depressed relative to non-reasoning models for that reason alone. Strip the block and re-score before comparing across model families.
* All 10 embedding models declared for VectorDB return identical retrieval metrics within every trial (and identical retrieved ids where checked), so the embedding-model sweep did not take effect (one Chroma collection was reused). The effective search space is far smaller than the declared one; drop the embedding factor from any configuration count.
* All rerankers tie on every metric at top_k 3, so the choice of `pass_reranker` is by cost, not by quality.
* The prompt-maker templates hard-code a question and do not use `{query}`, so the same fixed question was posed for every evaluation query; prompt-maker metrics cannot be interpreted as prompt quality.
* `context_length` in the generator module is the Ollama context window, not the chunk size; chunking is fixed at the data stage and was not varied inside these trials.