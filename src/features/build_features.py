import pathlib
import pandas as pd
import numpy as  np

from feature_definition import feature_build
from sklearn.model_selection import train_test_split


def load_data(data_path):
    df=pd.read_csv(data_path)
    return df



def save_data(train:pd.DataFrame,test:pd.DataFrame,output_path):
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    train.to_csv(output_path+'/train.csv',index=False)
    test.to_csv(output_path+'/test.csv',index=False)

if __name__=='__main__':
    train_data_path=pathlib.Path().cwd().as_posix()+'/data/raw/train.csv'
    test_data_path=pathlib.Path().cwd().as_posix()+'/data/raw/test.csv'
    output_path=pathlib.Path().cwd().as_posix()+'/data/processed'
    train_data=load_data(train_data_path)
    test_data=load_data(test_data_path)
    train_data=feature_build(train_data,'train-data')
    test_data=feature_build(test_data,'test-data')

    save_data(train=train_data,test=test_data,output_path=output_path)


    
    
    
    
