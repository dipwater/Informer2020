import pandas as pd


def filter_csv_columns(csv_file_path):
    """
    从指定的CSV文件中筛选出特定的列

    参数:
        csv_file_path (str): CSV文件的路径

    返回:
        pandas.DataFrame: 包含筛选后数据的DataFrame

    如果文件不存在或读取失败，返回None
    """

    # 需要保留的列
    selected_columns = [
        'date',
        'Actuator Z Position',
        'Motor Z Current',
        'Motor Y Temperature',
        'Motor Z Temperature',
        'Nut Y Temperature',
        'Ambient Temperature',
        'Motor Y Voltage'
    ]

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 筛选需要的列
        df_selected = df[selected_columns]

        return df_selected

    except FileNotFoundError:
        print(f"错误：找不到文件 '{csv_file_path}'")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


# 使用示例
if __name__ == "__main__":
    file_path = '../data/test/Spall.csv'
    result_df = filter_csv_columns(file_path)

    if result_df is not None:
        print("筛选成功！")
        print(result_df.head())

        # 可选：保存结果
        result_df.to_csv(file_path, index=False)
        print("已保存")
    else:
        print("筛选失败！")