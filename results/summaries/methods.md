# 统一结果（程序重算）

|方法|划分|运行数|Macro-F1均值|样本标准差|Accuracy均值|
|---|---|---:|---:|---:|---:|
|bert|test|3|0.915182|0.0034619822534572788|0.926212|
|tfidf|test|3|0.861040|0.0|0.884170|
|qwen_zero_shot|validation|1|0.662948|None|0.649485|
|qwen_few_shot|validation|1|0.753609|None|0.759021|
|qwen_lora|validation|3|0.911712|0.0018850612761014843|0.923540|
|qwen_zero_shot|test|1|0.625589|None|0.595882|
|qwen_few_shot|test|1|0.761049|None|0.765766|
|qwen_lora|test|3|0.911160|0.0014882772997754487|0.922780|
