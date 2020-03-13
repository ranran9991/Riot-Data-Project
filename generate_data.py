from riotwatcher import RiotWatcher, ApiError
import pandas as pd
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    REGIONS = ['ru', 'kr', 'br1', 'oc1', 'jp1', 'na1', 'eun1', 'euw1', 'tr1', 'la1', 'la2']
    parser = argparse.ArgumentParser(description='Get League of Legends match history data')
    parser.add_argument('--name',  help='Summoner name to get match history of', default='Vixerino')
    parser.add_argument('--region', help='Region of player', choices=REGIONS, default='eun1')
    parser.add_argument('--api_key', help='Riot games API key')
    parser.add_argument('--path', help='Path to save the output to', default='')
    args = parser.parse_args()

    watcher = RiotWatcher(args.api_key)

    summoner_name = args.name
    my_region = args.region

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
    matches_info_df.to_pickle(args.path + 'match_info.pkl')
