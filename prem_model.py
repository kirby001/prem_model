from understatapi import UnderstatClient
import pandas as pd
from scipy.stats import poisson
import multiprocessing
import datetime as dt
import numpy as np
import statsmodels.api as sm

FIXTURES_URL = 'https://fixturedownload.com/feed/json/epl-2024'

SEASONS = [
    '2014',
    '2015',
    '2016',
    '2017',
    '2018',
    '2019',
    '2020',
    '2021',
    '2022',
    '2023',
    '2024'
]


def create_fixtures_teams(url):
    replacements = [
        ('Spurs', 'Tottenham'), 
        ('Man Utd', 'Manchester United'), 
        ('Newcastle', 'Newcastle United'),
        ('Man City', 'Manchester City'),
        ('Wolves', 'Wolverhampton Wanderers'),
        ("Nott'm Forest", 'Nottingham Forest')
    ]
    fixtures = pd.read_json(url, orient='records')
    for pat, repl in replacements:
        fixtures['HomeTeam'] = fixtures['HomeTeam'].str.replace(pat=pat, repl=repl)
        fixtures['AwayTeam'] = fixtures['AwayTeam'].str.replace(pat=pat, repl=repl)
    all_teams = fixtures['HomeTeam'].unique().tolist()
    completed_fixtures = fixtures[~fixtures['HomeTeamScore'].isna()]
    upcoming_fixtures = fixtures[fixtures['HomeTeamScore'].isna()].sort_values(by=['DateUtc', 'MatchNumber'])
    upcoming_fixtures_list = upcoming_fixtures[['MatchNumber', 'HomeTeam', 'AwayTeam']].to_dict(orient='records')
    return all_teams, completed_fixtures, upcoming_fixtures_list


def create_team_record(team, completed_fixtures):
    team_record = {}
    team_record['name'] = team
    team_record['played'] = completed_fixtures[completed_fixtures['HomeTeam'] == team].shape[0] + completed_fixtures[completed_fixtures['AwayTeam'] == team].shape[0]
    team_record['wins'] = completed_fixtures.query(f'HomeTeam == "{team}" & HomeTeamScore > AwayTeamScore').shape[0] + completed_fixtures.query(f'AwayTeam == "{team}" & HomeTeamScore < AwayTeamScore').shape[0]
    team_record['draws'] = completed_fixtures.query(f'HomeTeam == "{team}" & HomeTeamScore == AwayTeamScore').shape[0] + completed_fixtures.query(f'AwayTeam == "{team}" & HomeTeamScore == AwayTeamScore').shape[0]
    team_record['losses'] = completed_fixtures.query(f'HomeTeam == "{team}" & HomeTeamScore < AwayTeamScore').shape[0] + completed_fixtures.query(f'AwayTeam == "{team}" & HomeTeamScore > AwayTeamScore').shape[0]
    team_record['goals_for'] = completed_fixtures[completed_fixtures['HomeTeam'] == team]['HomeTeamScore'].sum() + completed_fixtures[completed_fixtures['AwayTeam'] == team]['AwayTeamScore'].sum() 
    team_record['goals_against'] = completed_fixtures[completed_fixtures['HomeTeam'] == team]['AwayTeamScore'].sum() + completed_fixtures[completed_fixtures['AwayTeam'] == team]['HomeTeamScore'].sum()
    return team_record


def create_league_table(records):
    current_table_df = pd.DataFrame.from_records(records)
    current_table_df['points'] = 3 * current_table_df['wins'] + current_table_df['draws']
    current_table_df['goal_difference'] = current_table_df['goals_for'] - current_table_df['goals_against']
    current_table_df['max_points'] = (38 - current_table_df['played']) * 3 + current_table_df['points']
    current_table_df = current_table_df.sort_values(['points', 'goal_difference', 'name'], ascending=[False, False, True]).reset_index(drop=True)
    return current_table_df


def create_xG_table(league="EPL", season="2024"):
    understat = UnderstatClient()
    league_team_data = understat.league(league=league).get_team_data(season=season)
    all_teams_data = []
    for key in league_team_data.keys():
        team_data = {}
        team_name = league_team_data[key]['title']
        home_xG = 0.0
        home_xGA = 0.0
        home_played = 0
        away_xG = 0.0
        away_xGA = 0.0
        away_played = 0
        team_data['name'] = team_name
        for match in league_team_data[key]['history']:
            if match['h_a'] == 'h':
                home_xG += match['xG']
                home_xGA += match['xGA']
                home_played += 1
            elif match['h_a'] == 'a':
                away_xG += match['xG']
                away_xGA += match['xGA']
                away_played += 1
        team_data['home_xG_per_90'] = home_xG / home_played
        team_data['home_xGA_per_90'] = home_xGA / home_played
        team_data['away_xG_per_90'] = away_xG / away_played
        team_data['away_xGA_per_90'] = away_xGA / away_played
        all_teams_data.append(team_data)
    xG_df = pd.DataFrame.from_records(all_teams_data)
    return xG_df


def create_xG_table(league="EPL", season="2024"):
    understat = UnderstatClient()
    league_team_data = understat.league(league=league).get_team_data(season=season)
    all_teams_data = []
    for key in league_team_data.keys():
        team_data = {}
        team_name = league_team_data[key]['title']
        home_xG = 0.0
        home_xGA = 0.0
        home_played = 0
        away_xG = 0.0
        away_xGA = 0.0
        away_played = 0
        team_data['name'] = team_name
        for match in league_team_data[key]['history']:
            if match['h_a'] == 'h':
                home_xG += match['xG']
                home_xGA += match['xGA']
                home_played += 1
            elif match['h_a'] == 'a':
                away_xG += match['xG']
                away_xGA += match['xGA']
                away_played += 1
        team_data['home_xG_per_90'] = home_xG / home_played
        team_data['home_xGA_per_90'] = home_xGA / home_played
        team_data['away_xG_per_90'] = away_xG / away_played
        team_data['away_xGA_per_90'] = away_xGA / away_played
        all_teams_data.append(team_data)
    xG_df = pd.DataFrame.from_records(all_teams_data)
    return xG_df


def match_xG_data(xgdf, league='EPL', season='2024'):
    understat = UnderstatClient()
    league_match_data = understat.league(league=league).get_match_data(season=season)
    all_matches_data = []
    for match in league_match_data:
        if match['isResult']:
            match_data = {}
            match_data['home_team'] = match['h']['title']
            match_data['away_team'] = match['a']['title']
            match_data['home_goals'] = int(match['goals']['h'])
            match_data['away_goals'] = int(match['goals']['a'])
            match_data['home_xG'] = xgdf[xgdf['name'] == match_data['home_team']].iloc[0]['home_xG_per_90']
            match_data['away_xGA'] = xgdf[xgdf['name'] == match_data['away_team']].iloc[0]['away_xGA_per_90']
            match_data['away_xG'] = xgdf[xgdf['name'] == match_data['away_team']].iloc[0]['away_xG_per_90']
            match_data['home_xGA'] = xgdf[xgdf['name'] == match_data['home_team']].iloc[0]['home_xGA_per_90']
            all_matches_data.append(match_data)
    match_xG_df = pd.DataFrame.from_records(all_matches_data)
    return match_xG_df


def predict_home_goals(home_xG, away_xGA, params):
    return np.exp(params[0] + params[1] * home_xG + params[2] * away_xGA)


def predict_away_goals(away_xG, home_xGA, params):
    return np.exp(params[0] + params[1] * away_xG + params[2] * home_xGA)


def simulate_league(table, fixtures, xgdf, home_params, away_params):
    table = table.copy()
    for fixture in fixtures:
        home_mu = predict_home_goals(xgdf[xgdf['name'] == fixture['HomeTeam']].iloc[0]['home_xG_per_90'], xgdf[xgdf['name'] == fixture['AwayTeam']].iloc[0]['away_xGA_per_90'], home_params)
        away_mu = predict_away_goals(xgdf[xgdf['name'] == fixture['AwayTeam']].iloc[0]['away_xG_per_90'], xgdf[xgdf['name'] == fixture['HomeTeam']].iloc[0]['home_xGA_per_90'], away_params)
        home_goals = poisson.rvs(home_mu)
        away_goals = poisson.rvs(away_mu)
        table.loc[table['name'] == fixture['HomeTeam'], 'played'] += 1
        table.loc[table['name'] == fixture['AwayTeam'], 'played'] += 1
        table.loc[table['name'] == fixture['HomeTeam'], 'goals_for'] += home_goals
        table.loc[table['name'] == fixture['HomeTeam'], 'goals_against'] += away_goals
        table.loc[table['name'] == fixture['HomeTeam'], 'goal_difference'] += home_goals - away_goals
        table.loc[table['name'] == fixture['AwayTeam'], 'goals_for'] += away_goals
        table.loc[table['name'] == fixture['AwayTeam'], 'goals_against'] += home_goals
        table.loc[table['name'] == fixture['AwayTeam'], 'goal_difference'] += away_goals - home_goals
        if home_goals > away_goals:
            table.loc[table['name'] == fixture['HomeTeam'], 'wins'] += 1
            table.loc[table['name'] == fixture['HomeTeam'], 'points'] += 3
            table.loc[table['name'] == fixture['AwayTeam'], 'max_points'] -= 3
        elif home_goals == away_goals:
            table.loc[table['name'] == fixture['HomeTeam'], 'draws'] += 1
            table.loc[table['name'] == fixture['HomeTeam'], 'points'] += 1
            table.loc[table['name'] == fixture['HomeTeam'], 'max_points'] -= 2
            table.loc[table['name'] == fixture['AwayTeam'], 'draws'] += 1
            table.loc[table['name'] == fixture['AwayTeam'], 'points'] += 1
            table.loc[table['name'] == fixture['AwayTeam'], 'max_points'] -= 2
        elif home_goals < away_goals:
            table.loc[table['name'] == fixture['AwayTeam'], 'wins'] += 1
            table.loc[table['name'] == fixture['AwayTeam'], 'points'] += 3
            table.loc[table['name'] == fixture['HomeTeam'], 'max_points'] -= 3
        table.sort_values(['points', 'goal_difference', 'name'], ascending=[False, False, True], inplace=True)
        table.reset_index(drop=True, inplace=True)
        top_removed = table.iloc[1:]
        max_max_points = top_removed['max_points'].max()
        if ('match_won' not in table.columns) and table.loc[0, 'points'] > max_max_points:
            table['match_won'] = fixture['MatchNumber']
    if 'match_won' in table.columns:
        match_won = table.loc[0, 'match_won']
    else:
        match_won = 'final'
    outcome = {
        'winner': table.loc[0, 'name'],
        'winner_points': table.loc[0, 'points'],
        'second': table.loc[1, 'name'],
        'second_points': table.loc[1, 'points'],
        'match_won': match_won
    }
    return outcome


def run_multiple_simulations(number_of_simulations: int, current_table: pd.DataFrame, fixtures: list, xgdf: pd.DataFrame, home_params: list, away_params: list):
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    results = pool.starmap(simulate_league, [(current_table, fixtures, xgdf, home_params, away_params) for _ in range(number_of_simulations)])
    pool.close()
    pool.join()
    return results


def current_date_for_file():
    date = dt.datetime.now()
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute
    second = date.second
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}_{minute:02d}_{second:02d}"


if __name__ == '__main__':
    print(dt.datetime.now())
    all_teams, completed_fixtures, upcoming_fixtures = create_fixtures_teams(FIXTURES_URL)
    team_records = [create_team_record(team=team, completed_fixtures=completed_fixtures) for team in all_teams]
    current_league_table = create_league_table(team_records)
    xG_df = create_xG_table()
    dfs = []
    for season in SEASONS:
        xgdf = create_xG_table(season=season)
        matchxgdf = match_xG_data(xgdf=xgdf, season=season)
        dfs.append(matchxgdf)
    all_matches_df = pd.concat(dfs, ignore_index=True)
    df_home = all_matches_df[['home_xG', 'away_xGA', 'home_goals']]
    df_away = all_matches_df[['away_xG', 'home_xGA', 'away_goals']]
    X_home = df_home[['home_xG', 'away_xGA']]
    X_home = sm.add_constant(X_home)  # Add intercept
    y_home = df_home['home_goals']
    poisson_home = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    home_params = [poisson_home.params['const'], poisson_home.params['home_xG'], poisson_home.params['away_xGA']]
    X_away = df_away[['away_xG', 'home_xGA']]
    X_away = sm.add_constant(X_away)  # Add intercept
    y_away = df_away['away_goals']
    poisson_away = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
    away_params = [poisson_away.params['const'], poisson_away.params['away_xG'], poisson_away.params['home_xGA']]
    outcomes = run_multiple_simulations(100000, current_league_table, upcoming_fixtures, xG_df, home_params, away_params)
    outcome_df = pd.DataFrame.from_records(outcomes)
    outcome_df.to_csv(f'outcome_{current_date_for_file()}.csv')
    print(dt.datetime.now())
