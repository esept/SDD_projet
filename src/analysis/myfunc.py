import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

path = "../data_cleanup/clean_data.csv"
sns.set(style="whitegrid")  # 设置 Seaborn 图表样式

def process_time(df):
    df['created_time'] = pd.to_datetime(df['created_at'])
    df['YM'] = df['created_time'].dt.to_period('M').astype(str)  # 确保是字符串类型


def show_count_per_month(df, cols):
    plt.figure(figsize=(12, 7))
    for col in cols:
        data = df[df[col] == 1].groupby('YM').size().cumsum().reset_index(name='count')
        sns.lineplot(x='YM', y='count', data=data, marker='o', label=col)
        # 标记第一个和最后一个点
        for i in [0, len(data) - 1]:
            x = data.loc[i, 'YM']
            y = data.loc[i, 'count']
            plt.text(x, y, str(y), color='black', ha='center')
    plt.title("Cumulative Count per Month")
    plt.xlabel('Year-Month')
    plt.ylabel('Cumulative Count')
    plt.xticks(rotation=45)
    plt.legend(title='Series')
    plt.tight_layout()
    plt.show()

def show_sum_per_month(df, cols):
    plt.figure(figsize=(12, 7))
    for col in cols:
        data = df[df[col] == 1].groupby('YM')['downloads'].sum().cumsum().reset_index(name='sum')
        sns.lineplot(x='YM', y='sum', data=data, marker='o', label=col)
        # 标记第一个和最后一个点
        for i in [0, len(data) - 1]:
            x = data.loc[i, 'YM']
            y = data.loc[i, 'sum']
            plt.text(x, y, str(y), color='black', ha='center')
    plt.title("Cumulative Sum per Month")
    plt.xlabel('Year-Month')
    plt.ylabel('Cumulative Sum')
    plt.xticks(rotation=45)
    plt.legend(title='Series')
    plt.tight_layout()
    plt.show()


def show_count_per_month_subplot(df, cols):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=100)  # 增大宽度和DPI

    for idx, col in enumerate(cols):
        ax = axes[idx]  # 获取当前子图的轴
        data = df[df[col] == 1].groupby('YM').size().cumsum().reset_index(name='count')
        sns.lineplot(x='YM', y='count', data=data, marker='o', ax=ax)
        ax.set_title(col)
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('Cumulative Count')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def show_sum_per_month_subplot(df, cols):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=100)  # 增大宽度和DPI

    for idx, col in enumerate(cols):
        ax = axes[idx]  # 获取当前子图的轴
        data = df[df[col] == 1].groupby('YM')['downloads'].sum().cumsum().reset_index(name='sum')
        sns.lineplot(x='YM', y='sum', data=data, marker='o', ax=ax)
        ax.set_title(col)
        ax.set_xlabel('Year-Month')
        ax.set_ylabel('Cumulative Sum')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

# def show_count_per_month(df, cols):
#     plt.figure(1)
#     for col in cols:
#         data = df[df[col] == 1].groupby('YM').size().cumsum().reset_index(name='count')
#         data['YM'] = data['YM'].astype(str)  # 确保是字符串类型
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(x='YM', y='count', data=data, marker='o')
#         plt.title(f"{col} Count per Month")
#         plt.xlabel('Year-Month')
#         plt.ylabel('Cumulative Count')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()







def analyse_base_model(df):
    bm = df['base_models'].value_counts()
    print(len(bm))
    models_name = bm.index.tolist()
    models_nb = bm.values.tolist()
    print(len(models_name))
    bms = ['distilbert', 'xlm-roberta', 'roberta','bert', 'llama', 'gpt2', 'mistral', 'timm', 't5', 'marian', 'bart', 'whisper','gpt3']
    bmm = [[] for i in bms]
    nbbm = [0 for i in bms]
    for i in range(len(models_name)):
        ll = models_name[i].split(',')
        for j in ll:
            for m in range(len(bms)):
                if bms[m] in j:
                    bmm[m].append(i)
                    break
    # print(bmm)
    for i in range(len(bmm)):
        for j in range(len(bmm[i])):
            nbbm[i] += models_nb[j]
    print(nbbm)
    base_models_dict = {}
    for i,j in zip(bms,nbbm):
        base_models_dict[i] = j
    base_models_dict = dict(sorted(base_models_dict.items(),key = lambda x:x[1]))
    plt.bar(base_models_dict.keys(),base_models_dict.values())
    plt.xticks(rotation=25)
    plt.show()
    return base_models_dict






def run1():
    df = pd.read_csv(path)
    process_time(df)
    col_frameworks = ['framework_torch','framework_jax', 'framework_onnx', 'framework_tensorflow','framework_keras']
    show_count_per_month(df,col_frameworks)
    show_count_per_month_subplot(df,col_frameworks)

    show_sum_per_month(df,col_frameworks)
    show_sum_per_month_subplot(df, col_frameworks)
    col_ONEHOT = ['ONEHOT_endpoints_compatible', 'ONEHOT_autotrain_compatible','ONEHOT_safetensors', 'ONEHOT_tensorboard', 'ONEHOT_has_space']
    show_count_per_month(df,col_ONEHOT)
    show_sum_per_month(df,col_ONEHOT)
    show_count_per_month_subplot(df, col_ONEHOT)
    show_sum_per_month_subplot(df, col_ONEHOT)


if __name__ == "__main__":
    # run1()
    df = pd.read_csv(path)
    bmd = analyse_base_model(df)
    cols = ['created_at', 'downloads', 'likes','base_models']
    df = df[cols]
    process_time(df)
    bms = ['distilbert', 'xlm-roberta', 'roberta','bert', 'llama', 'gpt2', 'mistral', 'timm', 't5', 'marian', 'bart', 'whisper','gpt3']

    for i in bms:
        df[i] = 0
    for i in range(len(df)):
        bm_content = df.iloc[i,3]

        if type(bm_content) != str :
            continue
        ll = bm_content.split(',')
        for l in ll:
            for j in range(len(bms)):
                if bms[j] in l:
                    df.loc[i,bms[j]] = 1
                    break
    show_sum_per_month(df, bms)
    show_count_per_month(df, bms)
    show_sum_per_month_subplot(df, bms)
    show_count_per_month_subplot(df, bms)
