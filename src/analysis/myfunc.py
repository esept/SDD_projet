import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "../data_cleanup/clean_data.csv"

def process_time(df):
    df['created_time'] = pd.to_datetime(df['created_at'])
    df['YM'] = df['created_time'].dt.to_period('M')

def show_count_per_month(df,cols):
    for i in cols:
        model_month = df[df[i] == 1]['YM'].value_counts()
        modelss = model_month.cumsum()

        x_value = [str(p) for p in modelss.index]
        plt.plot(x_value,modelss.values.tolist())
        plt.title(i + "Count")
        plt.xlabel('Year_month')
        plt.xticks(rotation = 45)
        plt.show()

def show_sum_per_month(df,cols):
    for i in cols:
        model_month = df[df[i] == 1].groupby('YM')['downloads'].sum()
        modelss = model_month.cumsum()

        x_value = [str(p) for p in modelss.index]
        plt.plot(x_value,modelss.values.tolist())
        plt.title(i + "SUM")
        plt.xlabel('Year_month')
        plt.xticks(rotation = 45)
        plt.show()

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
    # plt.bar(base_models_dict.keys(),base_models_dict.values())
    # plt.xticks(rotation=25)
    # plt.show()
    return base_models_dict






def run1():
    df = pd.read_csv(path)
    process_time(df)
    # col_frameworks = ['framework_torch','framework_jax', 'framework_onnx', 'framework_tensorflow','framework_keras']
    # show_count_per_month(col_frameworks)
    # show_sum_per_month(col_frameworks)
    # col_ONEHOT = ['ONEHOT_endpoints_compatible', 'ONEHOT_autotrain_compatible','ONEHOT_safetensors', 'ONEHOT_tensorboard', 'ONEHOT_has_space']
    # show_count_per_month(col_ONEHOT)
    # show_sum_per_month(col_ONEHOT)


if __name__ == "__main__":
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
                    # print(bms[j])
                    df.loc[i,bms[j]] = 1
                    break
    # df.to_csv("./tt.csv")
    show_sum_per_month(df, bms)
    show_count_per_month(df, bms)
