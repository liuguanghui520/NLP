# 2026-10-25 Git提交范围

用户授权：提交推送。继续使用 `.publication/data-processing-v1` 中已有的 GitHub 仓库 `liuguanghui520/NLP` 和分支 `codex/data-processing-v1`，保留上游历史和课程材料。

本次同步项目根目录的 Markdown、环境检查与启动脚本、configs 和 scripts，以及 results 中不含评论正文的 JSON 配置、清单、汇总指标、状态和失败记录。原有公开的数据划分 ID、哈希与验收报告继续保留。

评论正文、逐样本 predictions.jsonl、原始数据、模型和适配器权重、SQLite 数据库、SwanLab 运行目录与运行日志保留本地，不纳入本次公开提交。API 运行目录因未含本次选取的 JSON 汇总而未同步。脚本及文档中的 qwen06-local-demo 是既有本机演示值，不是云端账户密钥。

校验：Python unittest 共 4 项通过（数据规则 2 项、指标逻辑 2 项）。首次测试遇到沙箱临时目录权限问题，改用仓库内被忽略的 logs/git-tests 临时目录后全部通过。未重新训练、评估模型或启动服务；这些是软件测试，不是新增模型质量结果。

AI 辅助：检查 Git 状态、筛选公开文件、补充忽略规则、运行已有轻量测试及执行提交推送。实验设计与结果解释未由本次操作生成。
