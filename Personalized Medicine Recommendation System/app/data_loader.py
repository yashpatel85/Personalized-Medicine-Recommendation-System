import pandas as pd

def load_and_merge_data(train_path='D:\Personalized Medicine Recommendation System\project\data\drugsComTest_raw.csv',
                        test_path='D:\Personalized Medicine Recommendation System\project\data\drugsComTrain_raw.csv') -> pd.DataFrame:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    df = pd.concat([train_df, test_df], ignore_index=True)
    df.dropna(subset=['review', 'condition', 'drugName', 'rating'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

