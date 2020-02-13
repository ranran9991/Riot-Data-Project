from riotwatcher import RiotWatcher, ApiError
import pandas as pd
from tqdm import tqdm

watcher = RiotWatcher('RGAPI-190e4f87-a1b4-4be3-8602-20461b167ada')

summoner_name = 'Vixerino'
my_region = 'eun1'

me = watcher.summoner.by_name(my_region, summoner_name)

# all objects are returned (by default) as a dict
# lets see if I got diamond yet (I probably didn't)
my_ranked_stats = watcher.league.by_summoner(my_region, me['id'])
matches = []
# get match outline loop
for start_index in range(0, 5000, 100):
    try:
        sub_matches = watcher.match.matchlist_by_account(my_region, me['accountId'], begin_index=start_index, end_index=start_index+100)['matches']
        matches += sub_matches
    except:
        break
matches = [match for match in matches if (match['queue'] == 400)]
matches_df = pd.DataFrame(matches)

# get more detailed match info loop
match_info = []
for i in tqdm(range(len(matches_df))):
    try:
        response = watcher.match.by_id(my_region, matches[i]['gameId'])
        match_info.append(response)
    except ApiError as err:
        if err.response.status_code == 429:
            response = watcher.match.by_id(my_region, matches[i]['gameId'])
            match_info.append(response)



matches_info_df = pd.DataFrame(match_info)
matches_info_df.to_pickle('match_info.pkl')
