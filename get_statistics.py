import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import ast

curr_account_id = '0VH_-CuMjRhsYkF9pOP9TXnRvZodh-rWWj22Ty0af2gpRw'

def only_dict(d):
    '''
    Convert json string representation of dictionary to a python dict
    '''
    return ast.literal_eval(d)

df = pd.read_pickle('match_info.pkl')
teams_df = pd.DataFrame(df.teams.tolist(), columns=['team_1', 'team_2'])
team_1 = json_normalize(teams_df['team_1']).add_prefix('team_1.')
team_2 = json_normalize(teams_df['team_2']).add_prefix('team_2.')

teams_df = team_1.join(team_2)

df = pd.concat([df, teams_df], sort=False, axis=1, ignore_index=False)
df = df.drop('teams', axis=1)

def get_player_num_by_id(row, id):
    for player in row:
        if player['player']['currentAccountId'] == id:
            return player['participantId']

df['participantIdentities'] = df['participantIdentities'].apply((lambda x: get_player_num_by_id(x, curr_account_id)))

def sort_out_teams(dataframe):
    cols = [col.split('.')[1] for col in list(df) if 'team_' in col]
    for col in cols:
        col1 = 'team_1.' + col
        col2 = 'team_2.' + col
        dataframe[col] = np.where(dataframe['participantIdentities']<5, dataframe[col1], dataframe[col2])

    for col in cols:
        col1 = 'team_1.' + col
        col2 = 'team_2.' + col
        try:
            dataframe.drop([col1, col2], axis=1, inplace=True)
        except:
            return


sort_out_teams(df)
df['participants'] = df.apply(lambda row: row.participants[row.participantIdentities-1], axis=1)
df['win'] = df['win'].replace(['Fail', 'Win'], [0, 1])
print(df['participants'].head(5))