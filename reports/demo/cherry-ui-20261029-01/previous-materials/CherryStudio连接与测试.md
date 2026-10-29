# Cherry Studio连接与测试

日期：2026-10-29。使用现有Qwen3-1.7B及T01适配器，不使用历史0.6B工程练习模型。先按[演示入口](README.md)启动项目服务。

## 客户端填写

在Cherry Studio的“设置→模型服务”增加一个OpenAI兼容的自定义服务；名称可填“NLP本地演示”。依据[官方服务商文档](https://github.com/CherryHQ/cherry-studio-docs/tree/main/pre-basic/providers)，具体字段位置以本机版本为准，不升级客户端或修改原有服务。

| 字段 | 填写内容 |
|---|---|
| 接口类型 | OpenAI兼容 Chat Completions |
| API地址 | `http://127.0.0.1:8017/v1` |
| API密钥 | `nlp-local-demo`（本机演示占位串，不是云端账号密钥） |
| 首选模型ID | `nlp-qwen-lora-t01-seed42` |
| 温度／输出上限 | 0／8；服务端仍固定贪心、最多8个新token |

服务支持`/v1/models`、`/v1/chat/completions`和SSE流式返回，也兼容客户端自动拼接时的`/models`、`/chat/completions`。最终请求应到`/v1/chat/completions`，避免重复成`/v1/v1/chat/completions`。手动加模型时使用下列准确ID，不能选云端同名Qwen：

```text
nlp-tfidf-seed42
nlp-macbert-seed42
nlp-qwen-zero-shot
nlp-qwen-four-shot
nlp-qwen-lora-t01-seed42
```

## 输入与结果

新建专用助手／话题，关闭联网、知识库和工具功能，仅粘贴评论正文，不加“判断情感”说明。服务端会按所选方案自动添加冻结提示；LoRA无示例，四样本使用训练集锁定的四条示例与原顺序。每条评论开新话题便于录屏；即使客户端带历史，服务端也只分类最后一条用户评论，并在日志记录忽略条数。

正常样例：

```text
房间干净，服务热情，下次还会入住。
```

本次五种方案对此输入均返回1。难例请从`configs/demo/cases.json`复制6353号完整评论，不能只复制关键片段冒充同一个输入。实测TF-IDF返回1，MacBERT、四样本和LoRA返回0；零样本输出包含解释，严格解析为非法。这些仅是本次少量演示结果，不用来重新计算总体准确率。

原始输出0为负面、1为正面；其余输出保留原文并标非法。API不添加解释或中文标签，方便与正式规则一致；中文标签、耗时和截断情况可在项目网页或请求JSONL查看。基线没有生成token计数，API usage中的0不当作实测token用量。

SSE在模型完成后发送原始输出和结束标记，已测试兼容协议；不是逐token到达速度测量。Cherry整次等待时间包含通信与可能的模型加载，不能拿来代替正式P50／P95。

## 当前连接状态

2026-10-29已确认Cherry安装路径存在；Windows启动工具等待应用授权超时，未获得可操作Cherry窗口。本地OpenAI兼容接口、五模型预测及SSE已通过独立请求检查；Cherry设置页面和发送按钮尚未实际验收。本人手动打开Cherry后，按上表配置并发送两条评论即可补齐客户端验收，保留真实聊天界面截图和对应request_id；不要写成已经通过。
