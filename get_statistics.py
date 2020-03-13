import pandas as pd
import numpy as np
from pandas import json_normalize
import ast
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

if __name__ == '__main__':
    curr_account_id = '0VH_-CuMjRhsYkF9pOP9TXnRvZodh-rWWj22Ty0af2gpRw'

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

    # take correct team according to player
    sort_out_teams(df)
    # take correct participant according to its ID
    df['participants'] = df.apply(lambda row: row.participants[row.participantIdentities-1], axis=1)
    df['win'] = df['win'].replace(['Fail', 'Win'], [0, 1])
    # champion Id column
    df['championId'] = df.apply(lambda row: row.participants['championId'], axis=1)
    df['stats'] = df.apply(lambda row: (row.participants['stats']), axis=1)
    df['lane'] = df.apply(lambda row: (row.participants['timeline']['lane']), axis=1)

    # remove useless columns
    df.drop(['participants', 'gameCreation', 'queueId', 'seasonId','gameId', 'bans', 'gameMode', 'gameType', 'gameVersion', 'mapId', 'participantIdentities', 'platformId', 'vilemawKills', 'dominionVictoryScore'], axis=1, inplace=True)

    stats_df = json_normalize(df['stats'].tolist()).drop(['inhibitorKills', 'win'], axis=1)
    # take meaningful columns from stats
    columns_to_keep = ['assists', 'champLevel', 'damageDealtToObjectives', 'damageDealtToTurrets', 'damageSelfMitigated', 'deaths', 'doubleKills', 'firstTowerKill', 'firstTowerAssist', 'goldEarned', 'goldSpent', 'killingSprees', 'kills', 'largestCriticalStrike', 'largestKillingSpree', 'largestMultiKill', 'longestTimeSpentLiving', 'magicDamageDealt', 'magicDamageDealtToChampions', 'magicalDamageTaken', 'neutralMinionsKilled', 'neutralMinionsKilledEnemyJungle', 'neutralMinionsKilledTeamJungle', 'pentaKills', 'physicalDamageDealt', 'physicalDamageDealtToChampions', 'physicalDamageTaken', 'totalDamageDealt', 'totalDamageDealtToChampions', 'totalDamageTaken', 'totalHeal', 'totalMinionsKilled', 'totalPlayerScore', 'totalTimeCrowdControlDealt', 'totalUnitsHealed', 'tripleKills', 'trueDamageDealt', 'trueDamageDealtToChampions', 'trueDamageTaken', 'turretKills', 'visionScore', 'wardsKilled', 'wardsPlaced']

    stats_df = stats_df[columns_to_keep]
    df = df.join(stats_df)
    df.drop(['stats', 'firstTowerKill', 'firstTowerAssist'], axis=1, inplace=True)
    print(f'Number of columns: {len(df.columns)}, Number of rows: {len(df.index)}')
    df.loc[:, df.columns != 'lane'] = df.loc[:, df.columns != 'lane'].astype(int)
    df = df.drop(['teamId'], axis=1)
    lane_df = df.groupby('lane')

    sns.set()
    def set_sizes(fig_size, font_size):
        plt.rcParams["figure.figsize"] = fig_size
        plt.rcParams["font.size"] = font_size
        plt.rcParams["xtick.labelsize"] = font_size
        plt.rcParams["ytick.labelsize"] = font_size
        plt.rcParams["axes.labelsize"] = font_size
        plt.rcParams["axes.titlesize"] = font_size
        plt.rcParams["legend.fontsize"] = font_size

    set_sizes((12,8), 10)
    LANES=['MIDDLE', 'TOP', 'JUNGLE', 'BOTTOM', 'NONE']
    # Games in each lane
    fig, ax = plt.subplots()
    ax.set_title('Number of games in each lane')
    ax.set_xlabel('lane')
    df['lane'].value_counts().plot(ax=ax, kind='bar')
    # Vision Score
    fig, ax = plt.subplots()
    ax.set_title('Vision Score in each lane')
    for lane in LANES:
        sns.distplot(df.loc[df['lane'] == lane]['visionScore'], label=lane, hist=False, kde=True)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Vision Score')
    ax.legend()
    # Duration
    fig, ax = plt.subplots()
    ax.set_title('Duration of games in each lane')
    df['gameDuration'] = df['gameDuration'].apply(lambda x: x/60)
    for lane in LANES:
        sns.distplot(df.loc[df['lane'] == lane]['gameDuration'], label=lane, hist=False, kde=True)
    ax.set_xlabel('Minutes')
    ax.set_ylabel('Frequency')
    ax.legend()
    # Stats
    fig, ax = plt.subplots()
    ax.set_title('Mean stats of games in each lane')
    lane_df.mean()[['kills', 'deaths', 'assists']].plot(ax=ax, kind='bar', width=0.45)
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    ax.legend()
    # Damage Dealt
    fig, ax = plt.subplots()
    ax.set_title('Mean Damage dealt to champions in each lane')
    lane_df.mean()[['trueDamageDealtToChampions', 'physicalDamageDealtToChampions', 'magicDamageDealtToChampions', 'totalDamageDealtToChampions']].plot(ax=ax, kind='bar', width=0.45)
    ax.legend(['True Damage Dealt To Champions', 'Physical Damage Dealt To Champions', 'Magic Damage Dealt To Champions', 'Total Damage Dealt To Champions'])
    # Damage Taken
    fig, ax = plt.subplots()
    ax.set_title('Mean Damage taken in each lane')
    lane_df.mean()[['trueDamageTaken', 'physicalDamageTaken', 'magicalDamageTaken', 'totalDamageTaken']].plot(ax=ax, kind='bar', width=0.45)
    ax.legend(['True Damage Taken', 'Physical Damage Taken', 'Magical Damage Taken', 'Total Damage Taken'])
    # Level
    fig, ax = plt.subplots()
    ax.set_title('Level in each lane')
    for lane in LANES:
        sns.distplot(df.loc[df['lane'] == lane]['champLevel'], label=lane, hist=False, kde=True)
    ax.set_xlabel('Level')
    ax.set_ylabel('Frequency')
    ax.legend()
    # Turret Kills
    fig, ax = plt.subplots()
    ax.set_title('Mean Turret Kills in each lane')
    for lane in LANES:
        sns.distplot(df.loc[df['lane'] == lane]['turretKills'], label=lane, hist=False, kde=True)
    ax.set_xlabel('Turret Kills')
    ax.set_ylabel('Frequency')
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title('Correlation between features')
    corr = df.drop(['lane'], axis=1).corr()
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(df.drop(['lane', 'championId'], axis=1).columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(df.drop(['lane', 'championId'], axis=1).columns)
    ax.set_yticklabels(df.drop(['lane', 'championId'],axis=1).columns)

    def multipage(filename, figs=None, dpi=200):
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
        pp.close()

    multipage('images.pdf')
    df.to_pickle('cleaned_data.pkl')
#plt.show()
