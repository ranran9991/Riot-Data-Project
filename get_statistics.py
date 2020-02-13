import pandas as pd
from pandas.io.json import json_normalize
import ast

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

print(teams_df.head(5))
df = pd.concat([df, teams_df], sort=False, axis=1, ignore_index=False)
df = df.drop('teams', axis=1)

    
print(df['participantIdentities'][0])