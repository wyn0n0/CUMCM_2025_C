import pandas as pd

# 读取文件
excel_file = pd.ExcelFile(r"G:\25国赛\C题\PythonProject\code2\noisy_data(15%).xlsx")

# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')

# 筛选出 Y 染色体浓度大于等于 0.04 的数据
filtered_df = df[df['Y染色体浓度加误差'] >= 0.04]

# 按照孕妇代码分组，找出检测孕期最小的行
result_df = filtered_df.loc[filtered_df.groupby('孕妇代码')['检测孕周'].idxmin()]

# 将结果保存为文件
result_path = '筛15%.xlsx'

result_df.to_excel(result_path, index=False)