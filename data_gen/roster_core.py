"""Transferable player state and lineup chemistry; organization is auxiliary only."""
from copy import deepcopy
from datetime import date
import itertools
import math
import numpy as np
from core import number, sigmoid
from data_gen.regional import RegionalRatings

ROLES = ['top', 'jng', 'mid', 'bot', 'sup']
DEFAULTS = {'history_length': 8, 'player_half_life_days': 365., 'form_half_life_days': 60.,
            'initial_variance': 1., 'minimum_variance': .08, 'team_aux_limit': .15}
STAT_NAMES = ['win', 'surprise', 'opponent', 'gd15', 'cs15', 'xp15', 'kp', 'deaths',
              'dpm', 'cspm', 'wpm', 'gold_known']
PLAYER_NAMES = ['ability', 'uncertainty', 'log_history', 'rest', 'known', 'role_fraction']+STAT_NAMES
HISTORY_NAMES = STAT_NAMES+['age', 'patch_match']
COHESION_NAMES = ['pair_games', 'pair_opponent_adjusted_form', 'five_player_games', 'five_player_form']
AUX_NAMES = ['team_recent_surprise', 'team_recent_gd15', 'team_history', 'lineup_retained']


def identity(name):
    value = (name or '').strip().casefold()
    return value if value not in {'', 'missing', 'unknown', 'nan'} else None


def canonical_roster(roster):
    return {role: identity(roster.get(role)) for role in ROLES}


def validate_roster(roster):
    if set(roster) != set(ROLES) or not all(isinstance(n, str) and identity(n) for n in roster.values()):
        raise ValueError('Supply five names under top,jng,mid,bot,sup.')
    normalized = canonical_roster(roster)
    if len(set(normalized.values())) != 5:
        raise ValueError('A lineup must contain five distinct players.')
    return normalized


def pair_key(names):
    return '\x1f'.join(sorted(names))


class RosterState:
    def __init__(self, regional_config=None, config=None):
        self.config = {**DEFAULTS, **(config or {})}
        if not isinstance(self.config['history_length'], int) or not 1 <= self.config['history_length'] <= 32:
            raise ValueError('history_length must be an integer between 1 and 32')
        for name in ['player_half_life_days', 'form_half_life_days', 'initial_variance', 'minimum_variance']:
            if not math.isfinite(self.config[name]) or self.config[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not 0 <= self.config['team_aux_limit'] <= .5:
            raise ValueError('team_aux_limit must be between zero and .5 logit')
        self.regional = RegionalRatings(regional_config)
        self.players, self.pairs, self.lineups, self.team_history, self.team_lineups = {}, {}, {}, {}, {}
        self.updated_games = 0

    def player(self, name, day):
        state = self.players.get(identity(name), {'ability': 0., 'variance': self.config['initial_variance'],
                                                  'n': 0, 'last_day': day, 'history': []})
        age = max(0, (date.fromisoformat(day)-date.fromisoformat(state['last_day'])).days)
        decay = 2**(-age/self.config['player_half_life_days'])
        return {**state, 'ability': state['ability']*decay,
                'variance': self.config['initial_variance']+(state['variance']-self.config['initial_variance'])*decay}

    def roster_strength(self, roster, day, region):
        roster = canonical_roster(roster)
        # Regional prior is match environment; learned abilities follow individuals
        # through roster changes and organization renames.
        regional = self.regional.log_strength(region, day)-self.regional.baseline(day)
        ability = sum(self.player(n, day)['ability'] for n in roster.values())/5
        return regional+ability

    def expected(self, a, b, day, region_a, region_b):
        return float(sigmoid(self.roster_strength(a, day, region_a)-self.roster_strength(b, day, region_b)))

    def arrays(self, roster, day, patch, team=None):
        roster = canonical_roster(roster)
        length = self.config['history_length']
        players = np.zeros((5, len(PLAYER_NAMES)), dtype=np.float32)
        history = np.zeros((5, length, len(HISTORY_NAMES)), dtype=np.float32)
        mask = np.zeros((5, length), dtype=bool)
        names = []
        for i, role in enumerate(ROLES):
            name = roster[role]
            names.append(name or '')
            state = self.player(name, day)
            rows = state['history'][-length:]
            ages = [max(0, (date.fromisoformat(day)-date.fromisoformat(row['day'])).days) for row in rows]
            weights = [2**(-age/self.config['form_half_life_days']) for age in ages]
            averages = [float(np.average([row[k] for row in rows], weights=weights)) if rows else (.5 if k == 'win' else 0.) for k in STAT_NAMES]
            rest = min(180, max(0, (date.fromisoformat(day)-date.fromisoformat(state['last_day'])).days))/90
            role_fraction = np.mean([row['role'] == role for row in rows]) if rows else 0.
            players[i] = [state['ability'], math.sqrt(state['variance']), math.log1p(state['n']),
                          rest, float(state['n'] > 0), role_fraction]+averages
            for j, (row, age) in enumerate(zip(rows, ages), length-len(rows)):
                history[i, j] = [row[k] for k in STAT_NAMES]+[min(180, age)/90, float(row['patch'] == patch)]
                mask[i, j] = True
        known = sorted(n for n in roster.values() if n)
        pairs = [self.pairs.get(pair_key(p), {'n': 0, 'surprise': 0.}) for p in itertools.combinations(known, 2)]
        full_key = pair_key(known) if len(known) == 5 else None
        full = self.lineups.get(full_key, {'n': 0, 'surprise': 0.})
        cohesion = np.array([math.log1p(np.mean([p['n'] for p in pairs])) if pairs else 0.,
                             np.mean([p['surprise']/(p['n']+5) for p in pairs]) if pairs else 0.,
                             math.log1p(full['n']), full['surprise']/(full['n']+5)], dtype=np.float32)
        team_rows = self.team_history.get(team, [])[-15:]
        last = set(n for n in self.team_lineups.get(team, {}).values() if n)
        retained = len(set(known) & last)/5
        auxiliary = np.array([np.mean([r['surprise'] for r in team_rows]) if team_rows else 0.,
                              np.mean([r['gd15'] for r in team_rows]) if team_rows else 0.,
                              math.log1p(len(team_rows)), retained], dtype=np.float32)
        return {'players': players, 'history': history, 'history_mask': mask, 'cohesion': cohesion,
                'auxiliary': auxiliary, 'player_names': names}

    def update(self, a, b, rows_a, rows_b, day, patch, league, team_a, team_b):
        a, b = canonical_roster(a), canonical_roster(b)
        region_a, region_b = self.regional.origins.get(team_a), self.regional.origins.get(team_b)
        p = self.expected(a, b, day, region_a, region_b)
        y = float(str(next(iter(rows_a.values()))['result']).lower() in {'true', '1'})
        # Diagonal approximate Bayesian logistic update. Uncertain/new players
        # receive more credit, and uncertainty contracts with observed games.
        states = [self.player(n, day) for n in [*a.values(), *b.values()]]
        information = p*(1-p)
        denominator = 1+information*sum(s['variance'] for s in states)/25
        opponent_a = self.roster_strength(b, day, region_b)
        opponent_b = self.roster_strength(a, day, region_a)
        for roster, rows, win, expected, opponent, team in [(a, rows_a, y, p, opponent_a, team_a),
                                                          (b, rows_b, 1-y, 1-p, opponent_b, team_b)]:
            surprise = win-expected
            kills = sum(number(row, 'kills') for row in rows.values())
            for role in ROLES:
                name, row = roster[role], rows.get(role)
                if not name or row is None:
                    continue
                state = self.player(name, day)
                variance = max(self.config['minimum_variance'], 1/(1/state['variance']+information/25))
                ability = state['ability']+state['variance']*.2*surprise/denominator
                known = bool(row.get('goldat15')) and number(row, 'goldat15') > 0
                record = {'day': day, 'patch': patch, 'role': role, 'win': win, 'surprise': surprise,
                          'opponent': opponent, 'gd15': number(row, 'golddiffat15')/1000 if known else 0.,
                          'cs15': number(row, 'csdiffat15')/20 if known else 0.,
                          'xp15': number(row, 'xpdiffat15')/1000 if known else 0.,
                          'kp': (number(row, 'kills')+number(row, 'assists'))/max(1, kills),
                          'deaths': number(row, 'deaths')/10, 'dpm': number(row, 'dpm')/1000,
                          'cspm': number(row, 'cspm')/10, 'wpm': number(row, 'wpm')/2,
                          'gold_known': float(known)}
                self.players[name] = {'ability': float(np.clip(ability, -4, 4)), 'variance': variance,
                                      'n': state['n']+1, 'last_day': day,
                                      'last_region': self.regional.origins.get(team),
                                      'history': (state['history']+[record])[-self.config['history_length']:]}
            names = sorted(n for n in roster.values() if n)
            for pair in itertools.combinations(names, 2):
                state = self.pairs.setdefault(pair_key(pair), {'n': 0, 'surprise': 0.})
                state['n'] += 1
                state['surprise'] += surprise
            if len(names) == 5:
                state = self.lineups.setdefault(pair_key(names), {'n': 0, 'surprise': 0.})
                state['n'] += 1
                state['surprise'] += surprise
            team_record = {'surprise': surprise, 'gd15': sum(number(r, 'golddiffat15') for r in rows.values())/5000}
            self.team_history[team] = (self.team_history.get(team, [])+[team_record])[-15:]
            self.team_lineups[team] = roster
        self.regional.update_regions(team_a, team_b, day, y, p, league)
        self.updated_games += 1

    def state(self):
        return deepcopy({'schema': 'player-core-v1', 'config': self.config, 'regional': self.regional.state(),
                         'players': self.players, 'pairs': self.pairs, 'lineups': self.lineups,
                         'team_history': self.team_history, 'team_lineups': self.team_lineups,
                         'updated_games': self.updated_games})

    @classmethod
    def from_state(cls, value):
        if value.get('schema') != 'player-core-v1':
            raise ValueError('Unsupported player-core state schema')
        state = cls(value['regional']['config'], value['config'])
        state.regional = RegionalRatings.from_state(value['regional'])
        for key in ['players', 'pairs', 'lineups', 'team_history', 'team_lineups', 'updated_games']:
            setattr(state, key, deepcopy(value[key]))
        return state
