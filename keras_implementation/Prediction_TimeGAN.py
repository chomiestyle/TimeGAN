import pandas as pd
from keras_implementation.TimeGan import TimeGAN


df_final = pd.read_csv('../Data/input/results_HD_new.csv')
df_final = df_final.tail(1825)
print(df_final)
step_size=5
num_feature=6
model=TimeGAN(step_size=step_size,feature_num=num_feature,df_final=df_final)
final_predictions=model.train()