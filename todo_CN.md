# Data Preprocess
- 将所有数据格式改为可用格式
- 清除所有空列

# Data Process
- 清理异常数据
- 分解数据格式(时间,…)

# Data Analyse
- 与语言的关系(langs,NB_langs,ONE-HOT_langs[NB_en,NB_ar,…最后六列])
- 与framework的关系(framework_…)
- 与论文数量(nb_arxiv),数据集数量(nb_dataset),…
- 与 基础模型的关系(base-models): 基础模型类似于开源软件,主要的问题是新模型的发布和维护
- 与 Task的关系(主要任务: pipeline_tag, 次要任务: tasks)
- ONE-HOT_…: 以下 tag 在统计中是出现次数最高的 tag`’ONEHOT_endpoints_compatible’.,ONEHOT_autotrain_compatibl','ONEHOT_safetensors', 'ONEHOT_tensorboard','ONEHOT_has_space'`

# Visualisation Application
使用`streamlit`实现一个及时的可视化页面