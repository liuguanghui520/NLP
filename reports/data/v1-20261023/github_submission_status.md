# GitHub提交状态

最新结果：已成功推送，远程分支 `codex/data-processing-v1`，提交 `5931fdf4a69da2aee0f7e173f5a19f208311927a`。下文认证失败为此前尝试的历史记录。通过Windows已启用的HTTP代理 `http://127.0.0.1:7897`，以单次 `git -c http.proxy=...` 完成推送，未修改全局Git配置。链接：https://github.com/liuguanghui520/NLP/tree/codex/data-processing-v1 。main未修改，未创建PR。

目标：https://github.com/liuguanghui520/NLP 。目标独立分支：codex/data-processing-v1，基于远程main；原有README和课程资料未修改。

2026-10-23：本地提交5931fdf已准备，共17个文件，位于项目目录 `.publication/data-processing-v1`。包含数据说明、处理运行说明、脚本、配置、统计核验报告、哈希与划分/子集ID，不含评论正文。

**尚未推送成功。** GitHub连接器create_branch返回403 Resource not accessible by integration；本机Git push返回Invalid username or token / Authentication failed。远程无本次提交，未创建PR。

补丁保存在本目录github-submission.patch，评论数据保留在原项目的data/processed/v1-20261023。无需重新转换。恢复GitHub写入认证后，在PowerShell执行：

```powershell
git -C 'C:\Users\lgh\Documents\ChatGPT\NLP\.publication\data-processing-v1' push -u origin codex/data-processing-v1
```

不要使用`git add .`将本地底稿、Alpaca、预检原文、复现副本或凭据补入公开提交。评论再分发许可仍待充分核验。此状态文件仅记录本地操作结果，不声称已公开发布。
