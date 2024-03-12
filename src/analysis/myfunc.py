import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "../data_cleanup/clean_data.csv"

def process_time(df):
    df['created_time'] = pd.to_datetime(df['created_at'])
    df['YM'] = df['created_time'].dt.to_period('M')

def show_count_per_month(cols):
    for i in cols:
        model_month = df[df[i] == 1]['YM'].value_counts()
        modelss = model_month.cumsum()

        x_value = [str(p) for p in modelss.index]
        plt.plot(x_value,modelss.values.tolist())
        plt.title(i + "Count")
        plt.xlabel('Year_month')
        plt.xticks(rotation = 45)
        plt.show()

def show_sum_per_month(cols):
    for i in cols:
        model_month = df[df[i] == 1].groupby('YM')['downloads'].sum()
        modelss = model_month.cumsum()

        x_value = [str(p) for p in modelss.index]
        plt.plot(x_value,modelss.values.tolist())
        plt.title(i + "SUM")
        plt.xlabel('Year_month')
        plt.xticks(rotation = 45)
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv(path)
    print(df.head())
    process_time(df)
    print(df.columns)
    col_frameworks = ['framework_torch','framework_jax', 'framework_onnx', 'framework_tensorflow','framework_keras']
    show_count_per_month(col_frameworks)
    show_sum_per_month(col_frameworks)
    col_ONEHOT = ['ONEHOT_endpoints_compatible', 'ONEHOT_autotrain_compatible','ONEHOT_safetensors', 'ONEHOT_tensorboard', 'ONEHOT_has_space']
    show_count_per_month(col_ONEHOT)
    show_sum_per_month(col_ONEHOT)

