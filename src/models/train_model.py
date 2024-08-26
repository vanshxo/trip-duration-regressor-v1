import pandas as pd
import pathlib
import sys
import joblib
import yaml

from sklearn.ensemble import RandomForestRegressor

def train_model(train_features,target,n_estimators,max_depth,seed):
    model=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,random_state=seed)
    model.fit(train_features,target)
    return model

def save_model(model,output_path):
    joblib.dump(model,output_path+'/model.joblib')


def main():
    data_path=pathlib.Path().cwd().as_posix()
    train_data=pd.read_csv(data_path+'/data/processed/train.csv')
    output_path=data_path+'/models'
    pathlib.Path(output_path).mkdir(parents=True,exist_ok=True)
    TARGET="trip_duration"
    params=yaml.load(open(data_path+'/params.yaml'),yaml.SafeLoader)['train_model']
    
    X = train_data.drop(TARGET, axis=1)
    y = train_data[TARGET]
    model=train_model(X,y,params['n_estimators'],params['max_depth'],params['seed'])
    save_model(model=model,output_path=output_path)

if __name__=='__main__':
    main()
