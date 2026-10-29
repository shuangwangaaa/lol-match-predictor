"""Causal hierarchical Elo with mutable, regularized region strength priors.

Region weights are positive and sum to one, both at initialization and after updates.
Domestic results update team effects. Only known cross-region games update region
effects; the probability and opponent strength are captured before every update.
"""
from copy import deepcopy
from datetime import date
import math
import numpy as np
from core import sigmoid

DEFAULTS = {
    'lck': .4, 'lpl': .3, 'lec': .1, 'others_total': .2,
    'regions': ['LCK', 'LPL', 'LEC', 'LCS', 'LCP', 'CBLOL'],
    'region_learning_rate': .06, 'region_half_life_days': 365.,
    'team_half_life_days': 365., 'team_shrink_per_game': .999,
    'region_max_log_deviation': math.log(4), 'form_half_life_days': 60.,
}
DOMESTIC_REGIONS = {'LCK': 'LCK', 'LCKC': 'LCK', 'LPL': 'LPL', 'LDL': 'LPL', 'LEC': 'LEC', 'LCS': 'LCS',
                    'LTA N': 'LCS', 'LTA S': 'CBLOL', 'CBLOL': 'CBLOL',
                    'NACL': 'LCS', 'CBLOLA': 'CBLOL',
                    'LCP': 'LCP', 'PCS': 'LCP', 'VCS': 'LCP', 'LJL': 'LCP', 'LCO': 'LCP'}
EUROPEAN_LEAGUES = {'LFL', 'LFL2', 'NLC', 'PRM', 'PRMP', 'LVP SL', 'LIT', 'EBL', 'LPLOL',
                    'UL', 'GLL', 'ESLOL', 'HLL', 'ROL', 'NEXO', 'HM', 'HC', 'TCL', 'AL'}
INTERNATIONAL = {'MSI', 'WCS', 'Worlds', 'FST', 'EWC'}
KS = np.array([8, 16, 24, 40, 64, 96], dtype=float)
ELO_PER_LOGIT = 400/math.log(10)
FEATURE_NAMES = ['region_log_strength', 'region_international_experience',
                 'opponent_adjusted_form_5', 'opponent_adjusted_form_15', 'opponent_adjusted_form_40',
                 'schedule_strength_5', 'schedule_strength_15', 'schedule_strength_40',
                 'schedule_adjusted_performance_5', 'schedule_adjusted_performance_15', 'schedule_adjusted_performance_40',
                 'expected_win_rate_5', 'expected_win_rate_15', 'expected_win_rate_40',
                 'region_known']


def region_for_league(league):
    from data_gen.deep_data import DOMESTIC
    league = league.removesuffix(' Promotion')
    if league in INTERNATIONAL or league == 'LTA':
        return None
    # A cup or regional multi-league event must not overwrite a home league.
    if league not in DOMESTIC:
        return None
    return DOMESTIC_REGIONS.get(league, 'LEC' if league in EUROPEAN_LEAGUES else 'DOM:'+league)


class RegionalRatings:
    def __init__(self, config=None):
        if config and 'other' in config:
            raise ValueError('Use others_total (shared mass), not the obsolete per-region other setting.')
        self.config = {**DEFAULTS, **(config or {})}
        for key in ['lck', 'lpl', 'lec', 'others_total', 'region_half_life_days', 'team_half_life_days', 'form_half_life_days']:
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f'regional.{key} must be finite and positive')
        if not 0 <= self.config['region_learning_rate'] <= 1:
            raise ValueError('region_learning_rate must be between zero and one')
        if not 0 < self.config['team_shrink_per_game'] <= 1 or self.config['region_max_log_deviation'] < 0:
            raise ValueError('Invalid hierarchical rating constraints')
        if not math.isclose(sum(self.config[k] for k in ['lck', 'lpl', 'lec', 'others_total']), 1., abs_tol=1e-10):
            raise ValueError('lck + lpl + lec + others_total must equal one')
        regions = self.config['regions']
        if not isinstance(regions, list) or not all(isinstance(r, str) and r for r in regions) or len(regions) != len(set(regions)):
            raise ValueError('regions must be a list of unique region names')
        if not {'LCK', 'LPL', 'LEC'} < set(regions):
            raise ValueError('regions must include LCK, LPL, LEC and at least one other region')
        self.config['regions'] = list(regions)
        self.origins, self.regions, self.teams = {}, {}, {}
        self.international_updates = 0
        self.unknown_international_games = 0
        self._weights_cache = None

    def prior(self, region):
        if region in {'LCK', 'LPL', 'LEC'}:
            return float(self.config[region.lower()])
        return float(self.config['others_total'])/(len(self.config['regions'])-3)

    def assign(self, team, league):
        region = region_for_league(league)
        # The registry is explicit and fixed before replay. Historical LLA is
        # merged into the current northern Americas group unless listed separately.
        if region == 'DOM:LLA':
            region = 'LLA' if 'LLA' in self.config['regions'] else 'LCS'
        if region not in self.config['regions']:
            region = None
        if region:
            old = self.origins.get(team)
            if old and old != region and team in self.teams:
                # Treat a competition-region change as a transfer, not free strength.
                self.teams[team]['local'] = [v*.5 for v in self.teams[team]['local']]
            self.origins[team] = region

    def _region(self, region, day):
        state = self.regions.get(region, {'delta': 0., 'last_day': day, 'games': 0})
        days = max(0, (date.fromisoformat(day)-date.fromisoformat(state['last_day'])).days)
        return {**state, 'delta': state['delta']*2**(-days/self.config['region_half_life_days'])}

    def _team(self, team, day):
        state = self.teams.get(team, {'local': [0.]*6, 'last_day': day, 'history': []})
        days = max(0, (date.fromisoformat(day)-date.fromisoformat(state['last_day'])).days)
        return {**state, 'local': np.array(state['local'])*2**(-days/self.config['team_half_life_days'])}

    def log_strength(self, region, day):
        weights = self.weights(day)
        if region not in weights:
            return self.baseline(day, weights)
        return math.log(weights[region])

    def weights(self, day):
        key = (day, self.international_updates)
        if self._weights_cache is not None and self._weights_cache[0] == key:
            return dict(self._weights_cache[1])
        names = self.config['regions']
        logits = np.array([math.log(self.prior(r))+self._region(r, day)['delta'] for r in names])
        values = np.exp(logits-logits.max())
        values /= values.sum()
        result = dict(zip(names, map(float, values)))
        self._weights_cache = (key, result)
        return dict(result)

    def baseline(self, day, weights=None):
        weights = self.weights(day) if weights is None else weights
        return float(np.mean([math.log(weights[r]) for r in self.config['regions'] if r not in {'LCK', 'LPL', 'LEC'}]))

    def ratings(self, team, day):
        region = self.origins.get(team)
        region_points = ELO_PER_LOGIT*(self.log_strength(region, day)-self.baseline(day))
        return 1500+region_points+self._team(team, day)['local']

    def pair(self, a, b, day):
        ea, eb = self.ratings(a, day), self.ratings(b, day)
        return ea, eb, sigmoid((ea-eb)/ELO_PER_LOGIT)

    def features(self, team, day):
        origin = self.origins.get(team)
        region = self._region(origin, day) if origin else {'games': 0}
        history = self._team(team, day)['history']
        result = [self.log_strength(origin, day)-self.baseline(day), math.log1p(region['games'])]
        # EWMAs use opponents' pre-game scores, never their future strength.
        for field, default in [('surprise', 0.), ('opponent', 0.), ('performance', 0.), ('expected', .5)]:
            for window in [5, 15, 40]:
                rows = history[-window:]
                weights = [2**(-max(0, (date.fromisoformat(day)-date.fromisoformat(r['day'])).days)
                              /self.config['form_half_life_days']) for r in rows]
                result.append(float(np.average([r[field] for r in rows], weights=weights)) if rows else default)
        return np.array(result+[float(origin is not None)], dtype=np.float32)

    def update(self, a, b, day, outcome, league):
        ea, eb, expected = self.pair(a, b, day)
        error = float(outcome)-expected
        for team, opponent, y, p, signed in [(a, eb, outcome, expected, error),
                                             (b, ea, 1-outcome, 1-expected, -error)]:
            state = self._team(team, day)
            history = list(state['history'])
            history.append({'day': day, 'surprise': float(y-p[2]),
                            'opponent': float((opponent[2]-1500)/400),
                            'performance': float((opponent[2]-1500)/400+2*y-1),
                            'expected': float(p[2])})
            self.teams[team] = {'local': (state['local']*self.config['team_shrink_per_game']+KS*signed).tolist(),
                                'last_day': day, 'history': history[-50:]}
        self.update_regions(a, b, day, outcome, float(expected[2]), league)

    def update_regions(self, a, b, day, outcome, expected_probability, league):
        """Update the normalized region layer using an externally supplied prior."""
        error = float(outcome)-float(expected_probability)
        ra, rb = self.origins.get(a), self.origins.get(b)
        if league in INTERNATIONAL and (not ra or not rb):
            self.unknown_international_games += 1
        if league in INTERNATIONAL and ra and rb and ra != rb:
            limit = self.config['region_max_log_deviation']
            for region, direction in [(ra, 1), (rb, -1)]:
                state = self._region(region, day)
                delta = state['delta']+direction*self.config['region_learning_rate']*error
                self.regions[region] = {'delta': float(np.clip(delta, -limit, limit)),
                                        'last_day': day, 'games': state['games']+1}
            self.international_updates += 1

    def coefficients(self, day):
        weights = self.weights(day)
        return {r: {'prior': self.prior(r), 'current': weights[r],
                    'international_games': self._region(r, day)['games']} for r in sorted(weights)}

    def state(self):
        return deepcopy({'weight_schema': 'simplex-v1', 'config': self.config, 'origins': self.origins, 'regions': self.regions,
                         'teams': self.teams, 'international_updates': self.international_updates,
                         'unknown_international_games': self.unknown_international_games})

    @classmethod
    def from_state(cls, state):
        if state.get('weight_schema') != 'simplex-v1':
            raise ValueError('Obsolete regional weight schema; rebuild features and retrain.')
        instance = cls(state['config'])
        for key in ['origins', 'regions', 'teams', 'international_updates', 'unknown_international_games']:
            setattr(instance, key, deepcopy(state[key]))
        return instance


def adjust_features(regional, a, b, day, d, player_elo=None, roster_a=None, roster_b=None):
    """Replace unanchored Elo; retain within-roster player contrasts with shrinkage."""
    d = np.array(d, dtype=np.float32, copy=True)
    ea, eb, _ = regional.pair(a, b, day)
    d[:6] = (ea-eb)/400
    d[6] = d[2]  # Series prior uses the same globally anchored team strength.
    if player_elo is not None and len(d) >= 78:
        strengths = []
        for team, roster in [(a, roster_a), (b, roster_b)]:
            names = [roster.get(r, '') for r in ['top', 'jng', 'mid', 'bot', 'sup']]
            known = [player_elo.get(p, 1500.) for p in names if p]
            mean = np.mean(known) if known else 1500.
            anchor = regional.ratings(team, day)[2]
            strengths.append(np.array([(anchor+.25*(player_elo.get(p, 1500.)-mean)-1500)/400 if p else 0. for p in names]))
        for i in range(5):
            d[47+i*5] = strengths[0][i]-strengths[1][i]
        d[74] = strengths[0].mean()-strengths[1].mean()
        d[75] = strengths[0].min()-strengths[1].min()
        # Old roster-strength-change was on an unanchored player-Elo scale.
        d[76] = 0.
    return np.r_[d, regional.features(a, day)-regional.features(b, day)].astype(np.float32)
