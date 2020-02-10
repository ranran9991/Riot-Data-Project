from riotwatcher import RiotWatcher, ApiError
import pandas as pd

watcher = RiotWatcher('RGAPI-190e4f87-a1b4-4be3-8602-20461b167ada')

summoner_name = 'Vixerino'
my_region = 'eun1'

me = watcher.summoner.by_name(my_region, summoner_name)

# all objects are returned (by default) as a dict
# lets see if I got diamond yet (I probably didn't)
my_ranked_stats = watcher.league.by_summoner(my_region, me['id'])
matches = []
for start_index in range(0, 5000, 100):
    try:
        sub_matches = watcher.match.matchlist_by_account(my_region, me['accountId'], begin_index=start_index, end_index=start_index+100)['matches']
        matches += sub_matches
    except:
        break

matches = [match for match in matches if (match['queue'] == 400)]
df = pd.DataFrame(matches)
print(df.head(5))
print(len(df))
# For Riot's API, the 404 status code indicates that the requested data wasn't found and
# should be expected to occur in normal operation, as in the case of a an
# invalid summoner name, match ID, etc.
#
# The 429 status code indicates that the user has sent too many requests
# in a given amount of time ("rate limiting").

try:
    response = watcher.summoner.by_name(my_region, summoner_name)
except ApiError as err:
    if err.response.status_code == 429:
        print('We should retry in {} seconds.'.format(err.response.headers['Retry-After']))
        print('this retry-after is handled by default by the RiotWatcher library')
        print('future requests wait until the retry-after time passes')
    elif err.response.status_code == 404:
        print('Summoner with that ridiculous name not found.')
    else:
        raise