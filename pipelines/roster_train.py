"""Train, calibrate and forecast with players as the primary statistical unit."""
import csv
import hashlib
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from core import ROOT, metrics, sigmoid, logit
from data_gen.regional import DEFAULTS as REGIONAL_DEFAULTS
from data_gen.roster_core import DEFAULTS, ROLES, RosterState, validate_roster
from data_gen.roster_data import ARRAY_FIELDS, build, match_input
from models.roster_models import INPUT_KEYS, RosterTransformer

OUT = ROOT/'artifacts_roster'


def signature(config):
    files = ['data/teams_all.csv', 'data/players.csv.gz', 'data_gen/roster_core.py',
             'data_gen/roster_data.py', 'data_gen/regional.py', 'models/roster_models.py']
    return {'regional': {**REGIONAL_DEFAULTS, **config.get('regional', {})},
            'players': {**DEFAULTS, **config.get('players', {})},
            'sources': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}


def prepare(config):
    OUT.mkdir(exist_ok=True)
    if not (ROOT/'data'/'players.csv.gz').exists():
        raise FileNotFoundError('Player-core training requires data/players.csv.gz')
    from data_gen.deep_data import write_all_teams
    write_all_teams()
    fingerprint = signature(config)
    if (OUT/'preparation.json').exists() and json.loads((OUT/'preparation.json').read_text(encoding='utf-8')) == fingerprint:
        with np.load(OUT/'data.npz', allow_pickle=False) as stream:
            raw = {key: stream[key] for key in stream.files}
        return raw, json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    samples, metadata = build(config.get('regional'), config.get('players'), save_snapshot=OUT/'snapshot.json')
    raw = {k: np.stack([s[k] for s in samples]) for k in ARRAY_FIELDS}
    raw.update(y=np.array([s['y'] for s in samples], dtype='float32'),
               map_target=np.array([s['map_fraction'] for s in samples], dtype='float32'),
               game_count=np.array([s['game_count'] for s in samples], dtype='float32'),
               best_of=np.array([s['best_of'] for s in samples], dtype='int64'))
    rows = [{k: v for k, v in s.items() if k not in ARRAY_FIELDS} for s in samples]
    np.savez_compressed(OUT/'data.npz', **raw)
    for name, value in [('samples', rows), ('feature_metadata', metadata), ('preparation', fingerprint)]:
        (OUT/f'{name}.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'prepared': len(rows), 'players': metadata['players'], 'updated_games': metadata['updated_games'],
                      'missing_target_player_slots': metadata['missing_target_player_slots']}), flush=True)
    return raw, rows


class PlayerNormalizer:
    def fit(self, raw, ids, samples, fine_ids=()):
        self.stats = {}
        for field in ['players', 'history', 'cohesion', 'aux', 'context']:
            if field == 'context':
                values = raw['context'][ids]
            else:
                values = []
                for side in ['a', 'b']:
                    x = raw[field+'_'+side][ids]
                    values.append(x[raw['mask_'+side][ids]] if field == 'history' else x.reshape(-1, x.shape[-1]))
                values = np.concatenate(values)
            if not len(values):
                raise ValueError(f'No usable {field} observations in the training partition.')
            mean, scale = values.mean(0), np.maximum(values.std(0), .1)
            if field == 'aux':
                mean[3], scale[3] = 0., 1.  # Preserve the roster-retention gate.
            self.stats[field] = {'mean': mean.tolist(), 'scale': scale.tolist()}
        # Fine-tuning names are known only from 2024/25 Worlds, not 2026 future lineups.
        vocabulary_ids = np.r_[ids, np.asarray(fine_ids, dtype=int)]
        names = sorted({n for i in vocabulary_ids for side in ['a', 'b'] for n in samples[i]['names_'+side] if n})
        self.players = {n: i+1 for i, n in enumerate(names)}
        leagues = sorted({samples[i]['league'] for i in ids} | {'WCS'})
        self.leagues = {name: i+1 for i, name in enumerate(leagues)}
        return self

    def transform(self, raw, samples, device):
        arrays = {k: np.array(v) for k, v in raw.items()}
        for field in self.stats:
            fields = ['context'] if field == 'context' else [field+'_a', field+'_b']
            mean, scale = np.array(self.stats[field]['mean']), np.array(self.stats[field]['scale'])
            for name in fields:
                arrays[name] = np.clip((raw[name]-mean)/scale, -7, 7).astype('float32')
                if field == 'history':
                    arrays[name][~raw['mask_'+name[-1]]] = 0.
        for side in ['a', 'b']:
            arrays['ids_'+side] = np.array([[self.players.get(n, 0) for n in row['names_'+side]] for row in samples], dtype='int64')
        arrays['league'] = np.array([self.leagues.get(row['league'], 0) for row in samples], dtype='int64')
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}

    def state(self):
        return {'stats': self.stats, 'players': self.players, 'leagues': self.leagues}

    @classmethod
    def from_state(cls, value):
        result = cls()
        result.stats, result.players, result.leagues = value['stats'], value['players'], value['leagues']
        return result


def inputs(tensors, ids):
    return [tensors[k][ids] for k in INPUT_KEYS]


@torch.no_grad()
def infer(model, tensors, ids, player_only=False, return_components=False):
    model.eval()
    result, core, auxiliary = [], [], []
    for start in range(0, len(ids), 128):
        take = ids[start:start+128]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tensors['offset'].is_cuda):
            parts = model.decompose(*inputs(tensors, take))
            z = parts['player_prior']+parts['player_correction'] if player_only else parts['logit']
        result.append(z.float().cpu().numpy())
        if return_components:
            core.append((parts['player_prior']+parts['player_correction']).float().cpu().numpy())
            auxiliary.append(parts['team_auxiliary'].float().cpu().numpy())
    return (np.concatenate(result), np.concatenate(core), np.concatenate(auxiliary)) if return_components else np.concatenate(result)


def loss(model, tensors, ids):
    from models.deep_models import series_logits
    parts = model.decompose(*inputs(tensors, ids))
    z = parts['logit'].float()
    per_map = F.binary_cross_entropy_with_logits(z, tensors['map_target'][ids], reduction='none')
    counts = tensors['game_count'][ids]
    map_loss = (per_map*counts).sum()/counts.sum()
    trusted = tensors['best_of'][ids] >= 3
    series_loss = F.binary_cross_entropy_with_logits(series_logits(z[trusted], tensors['best_of'][ids][trusted]),
                                                    tensors['y'][ids][trusted]) if trusted.any() else map_loss
    return .5*map_loss+.5*series_loss+.02*parts['team_auxiliary'].square().mean()


def fit(model, tensors, ids, val, stage, epochs, fixed=False):
    fine = stage == 'worlds_finetune'
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5 if fine else 2e-4, weight_decay=.03)
    best, best_epoch, best_state, trace = float('inf'), 0, None, []
    ids = torch.as_tensor(ids, device=tensors['offset'].device)
    for epoch in range(1, epochs+1):
        model.train()
        order = ids[torch.randperm(len(ids), device=ids.device)]
        values = []
        for start in range(0, len(ids), 128):
            batch = order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=ids.is_cuda):
                value = loss(model, tensors, batch)
            if not torch.isfinite(value):
                raise RuntimeError('Non-finite roster-model loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            values.append(float(value.detach()))
        if val is not None:
            model.eval()
            with torch.no_grad():
                batches = [val[i:i+256] for i in range(0, len(val), 256)]
                score = sum(float(loss(model, tensors, batch))*len(batch) for batch in batches)/len(val)
        else:
            score = float(np.mean(values))
        trace.append({'epoch': epoch, 'train_loss': float(np.mean(values)), 'validation_loss': score if val is not None else None})
        if not fixed and score < best-1e-5:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({'stage': stage, **trace[-1]}), flush=True)
        if not fixed and epoch-best_epoch >= 4:
            break
    if not fixed:
        model.load_state_dict(best_state)
    return {'epochs': epochs if fixed else best_epoch, 'trace': trace}


def run(config):
    from main import seed_everything, raw_series
    from pipelines.optimize import calibrate, apply_calibration
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw, samples = prepare(config)
    dates = np.array([s['date'] for s in samples])
    ends = np.array([s['end_date'] for s in samples])
    domestic = np.array([s['domain'] == 'domestic' for s in samples])
    years = np.array([s['worlds_year'] for s in samples])
    train_select = np.flatnonzero(domestic & (ends < '2025-07-01'))
    val_select = np.flatnonzero(domestic & (dates >= '2025-07-01') & (ends < '2025-10-14'))
    fine_select = np.flatnonzero(years == '2024')
    fine_val = np.flatnonzero(years == '2025')
    train_ids = np.flatnonzero(domestic & (ends < '2026-04-01'))
    fine_ids = np.flatnonzero((years == '2024') | (years == '2025'))
    major = {s['id'] for s in json.loads((ROOT/'artifacts_optimized'/'samples.json').read_text(encoding='utf-8'))}
    trusted = raw['best_of'] >= 3
    major_mask = np.array([s['id'] in major for s in samples])
    early = np.flatnonzero(major_mask & trusted & (dates >= '2026-04-01') & (ends < '2026-05-01'))
    late = np.flatnonzero(major_mask & trusted & (dates >= '2026-05-01') & (ends < '2026-05-24'))
    calibration = np.r_[early, late]
    protocol = {'feature_signature': signature(config), 'seeds': config['seeds'],
                'domestic_pretraining_n': len(train_ids), 'domestic_end_exclusive': '2026-04-01',
                'worlds_finetuning_n': len(fine_ids), 'worlds_finetune_years': [2024, 2025],
                'worlds_2026_optimizer_labels_used': 0, 'calibration_n': len(calibration),
                'calibration_validation_end_exclusive': '2026-05-24',
                'selection_max_epochs': [config['max_pretrain_epochs'], config['max_finetune_epochs']],
                'unit': 'Player identities, role-specific histories and player-pair/five-player coappearance; no team-ID embedding',
                'team_context': 'Bounded auxiliary branch gated by current-roster retention',
                'status': 'Previously inspected retrospective benchmark; design follows earlier IG-LYON diagnosis.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    all_p, core_p, traces, choices = [], [], {}, None
    for seed in config['seeds']:
        seed_everything(seed)
        if choices is None:
            norm = PlayerNormalizer().fit(raw, train_select, samples, fine_select)
            tensors = norm.transform(raw, samples, device)
            dimensions = {'players': len(norm.players)+1, 'leagues': len(norm.leagues)+1,
                          'history_length': raw['history_a'].shape[2],
                          'team_aux_limit': {**DEFAULTS, **config.get('players', {})}['team_aux_limit']}
            model = RosterTransformer(**dimensions).to(device)
            pre = fit(model, tensors, train_select, val_select, 'player_selection_pretrain', config['max_pretrain_epochs'])
            fine = fit(model, tensors, fine_select, fine_val, 'worlds_finetune', config['max_finetune_epochs'])
            choices = {'pretrain_epochs': pre['epochs'], 'finetune_epochs': fine['epochs'], 'selection_seed': seed}
            traces['selection'] = [pre, fine]
            (OUT/'choices.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
            del model, tensors
        seed_everything(seed)
        norm = PlayerNormalizer().fit(raw, train_ids, samples, fine_ids)
        tensors = norm.transform(raw, samples, device)
        dimensions = {'players': len(norm.players)+1, 'leagues': len(norm.leagues)+1,
                      'history_length': raw['history_a'].shape[2],
                      'team_aux_limit': {**DEFAULTS, **config.get('players', {})}['team_aux_limit']}
        model = RosterTransformer(**dimensions).to(device)
        pre = fit(model, tensors, train_ids, None, 'player_final_pretrain', choices['pretrain_epochs'], fixed=True)
        folder = OUT/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        def save(stage):
            torch.save({'config': dimensions, 'state_dict': model.state_dict(), 'normalizer': norm.state(),
                        'feature_signature': protocol['feature_signature'], 'stage': stage}, folder/f'{stage}.pt')
        save('pretrained')
        fine = fit(model, tensors, fine_ids, None, 'worlds_finetune', choices['finetune_epochs'], fixed=True)
        save('finetuned')
        logits, core_logits, auxiliary = infer(model, tensors, np.arange(len(samples)), return_components=True)
        all_p.append(sigmoid(raw_series(logits, raw['best_of'])))
        core_p.append(sigmoid(raw_series(core_logits, raw['best_of'])))
        np.savez_compressed(folder/'predictions.npz', game_logits=logits, core_game_logits=core_logits,
                            auxiliary_game_logits=auxiliary, series_probability=all_p[-1])
        traces[str(seed)] = [pre, fine]
        print(json.dumps({'seed_complete': seed, 'parameters': sum(p.numel() for p in model.parameters())}), flush=True)
        del model, tensors
    deep = logit(np.mean(all_p, axis=0))
    core = logit(np.mean(core_p, axis=0))
    prior = raw_series(raw['offset'], raw['best_of'])
    early_weights = calibrate(deep[early], prior[early], raw['y'][early])
    validation = metrics(raw['y'][late], apply_calibration(deep[late], prior[late], early_weights))
    selection = {'choices': choices, 'forward_validation': validation, 'validation_end_exclusive': '2026-05-24',
                 'benchmark_seen_for_selection': False}
    (OUT/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    weights = calibrate(deep[calibration], prior[calibration], raw['y'][calibration])
    probability = apply_calibration(deep, prior, weights)
    player_only = apply_calibration(core, prior, weights)
    original = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    benchmark = np.array([i for i, row in enumerate(samples) if row['id'] in original])
    assert len(benchmark) == 412 and all(int(raw['y'][i]) == int(original[samples[i]['id']]['y']) for i in benchmark)
    world_ids = np.flatnonzero((years == '2026') & trusted)
    international = np.flatnonzero(major_mask & trusted & ~domestic & (dates >= '2026-05-24'))
    results = {name: {'reused_412': metrics(raw['y'][benchmark], p[benchmark]),
                      'worlds_2026_series': metrics(raw['y'][world_ids], p[world_ids]),
                      'international_since_2026_05_24': metrics(raw['y'][international], p[international])}
               for name, p in [('roster_transformer', probability), ('player_only_ablation', player_only)]}
    report = {'protocol': protocol, 'selection': selection, 'calibration_weights': weights.tolist(), 'results': results,
              'ablation': 'Same weights/calibrator, organization auxiliary set to zero; no refitting on test'}
    (OUT/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(traces, indent=2), encoding='utf-8')
    np.savez_compressed(OUT/'ensemble_predictions.npz', deep=deep, core=core, prior=prior, probability=probability)
    rows = [{**{k: samples[i][k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']},
             'probability': float(probability[i]), 'player_only': float(player_only[i])} for i in benchmark]
    with (OUT/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(report)
    print('ROSTER_RESULTS', json.dumps({name: {task: {k: v for k, v in score.items() if k != 'calibration_bins'}
                                              for task, score in value.items()} for name, value in results.items()}), flush=True)
    return report


def forecast(a_name, b_name, day, patch, best_of, league='WCS', playoffs=0,
             roster_a=None, roster_b=None, region_a=None, region_b=None, *, artifact_dir=None,
             model_name='roster_transformer'):
    from collections import Counter
    from main import raw_series
    from pipelines.optimize import apply_calibration
    torch.set_num_threads(4)
    folder = OUT if artifact_dir is None else artifact_dir
    report = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    snapshot = json.loads((folder/'snapshot.json').read_text(encoding='utf-8'))
    if day <= snapshot['snapshot_last_day']:
        raise ValueError('Snapshot includes this date or later; use stored historical predictions.')
    state = RosterState.from_state(snapshot['state'])
    expected = report['protocol']['feature_signature']
    if state.config != expected['players'] or state.regional.config != expected['regional']:
        raise ValueError('Snapshot and calibrated model configurations differ; retrain.')
    aliases = {n.casefold(): key for n, key in snapshot['aliases'].items()}
    a, b = aliases.get(a_name.casefold(), a_name), aliases.get(b_name.casefold(), b_name)
    if a == b:
        raise ValueError('Choose distinct sides.')
    def resolve(team, supplied, explicit_region):
        roster = state.team_lineups.get(team) if supplied is None else supplied
        if roster is None:
            raise ValueError('New organizations need an explicit five-player lineup.')
        if not isinstance(roster, dict):
            raise ValueError('A lineup must be a JSON object mapping top,jng,mid,bot,sup to names.')
        roster = validate_roster(roster)
        region = explicit_region or state.regional.origins.get(team)
        if region is None:
            known = [state.players.get(n, {}).get('last_region') for n in roster.values()]
            known = [r for r in known if r]
            region = Counter(known).most_common(1)[0][0] if known else None
        if region not in state.regional.config['regions']:
            raise ValueError('Supply a known region for an unrecognized organization/lineup.')
        return roster, region
    ra, region_a = resolve(a, roster_a, region_a)
    rb, region_b = resolve(b, roster_b, region_b)
    if set(ra.values()) & set(rb.values()):
        raise ValueError('A player cannot appear on both sides of a match.')
    row = match_input(state, a, b, ra, rb, day, patch, best_of, league, playoffs, region_a, region_b)
    raw = {k: np.asarray(row[k])[None] for k in ARRAY_FIELDS}
    sample = {'league': league, 'names_a': row['names_a'], 'names_b': row['names_b']}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p, core, decompositions = [], [], []
    stage = report['protocol'].get('checkpoint_stage', 'finetuned')
    for seed in report['protocol']['seeds']:
        checkpoint = torch.load(folder/f'seed{seed}'/f'{stage}.pt', map_location='cpu', weights_only=True)
        if checkpoint['feature_signature'] != expected or checkpoint['stage'] != stage:
            raise ValueError('Checkpoint does not match the calibrated player model; retrain.')
        norm = PlayerNormalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, [sample], device)
        model = RosterTransformer(**checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        total, player, auxiliary = infer(model, tensors, np.array([0]), return_components=True)
        p.append(float(sigmoid(raw_series(total, np.array([best_of])))[0]))
        core.append(float(sigmoid(raw_series(player, np.array([best_of])))[0]))
        decompositions.append({'seed': seed, 'player_prior_logit': float(raw['offset'][0]),
                               'player_network_correction_logit': float(player[0]-raw['offset'][0]),
                               'team_auxiliary_logit': float(auxiliary[0])})
    prior = raw_series(raw['offset'], np.array([best_of]))
    value = float(apply_calibration(np.array([logit(np.mean(p))]), prior, report['calibration_weights'])[0])
    no_team = float(apply_calibration(np.array([logit(np.mean(core))]), prior, report['calibration_weights'])[0])
    return {'model': model_name, 'team_a': a_name, 'team_b': b_name, 'date': day, 'patch': patch,
            'best_of': best_of, 'p_a_win': value, 'p_b_win': 1-value, 'player_only_p_a_win': no_team,
            'lineup_a': ra, 'lineup_b': rb, 'regions': [region_a, region_b],
            'region_weights': state.regional.weights(day), 'snapshot_last_day': snapshot['snapshot_last_day'],
            'history_since': snapshot.get('history_since'),
            'seed_decompositions': decompositions,
            'lineup_status': 'Supplied or last-known lineup; current official starters not fetched.'}


def test(args, config, parser, *, artifact_dir=None, model_name='roster_transformer'):
    try:
        folder = OUT if artifact_dir is None else artifact_dir
        report = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        expected = report['protocol']['feature_signature']
        if {**REGIONAL_DEFAULTS, **config.get('regional', {})} != expected['regional'] or {**DEFAULTS, **config.get('players', {})} != expected['players']:
            raise ValueError('Feature configuration changed; retrain before prediction.')
        stage = report['protocol'].get('checkpoint_stage', 'finetuned')
        if sorted(config['seeds']) != sorted(report['protocol']['seeds']) or config['stage'] != stage:
            raise ValueError(f'Use the saved seeds and {stage} stage for this calibrator.')
        output = forecast(args.team_a, args.team_b, args.date, args.patch, args.best_of, args.league, args.playoffs,
                           json.loads(args.roster_a) if args.roster_a else None,
                           json.loads(args.roster_b) if args.roster_b else None, args.region_a, args.region_b,
                           artifact_dir=folder, model_name=model_name)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(output, indent=2, ensure_ascii=False))


def write_report(report):
    metadata = json.loads((OUT/'feature_metadata.json').read_text(encoding='utf-8'))
    results = report['results']
    lines = ['# 以选手与实际阵容为核心的预测模型', '',
             '选手状态按个人保存，换队不清零。队名只用于查找已知首发、赛区与有上限的组织辅助历史，没有队名或战队 ID embedding。', '',
             '## 核心表示', '',
             '每名选手包含个人贝叶斯近似动态能力、估计不确定性、位置适配与近期对手修正统计；最近 8 局个人历史由共享时间 Transformer 编码，再通过五位置阵容 Transformer 和 Dense 分支聚合。', '',
             '选手 embedding 字典只来自对应训练期与历史世界赛微调期；未来未见选手使用 UNK 与连续历史特征。个人能力更新使用全局阵容赛前预期，统计对手强度在当时冻结。', '',
             '双人和完整五人共同出场经验按选手组合保存，与组织名称无关；大换血会切换阵容组合。组织辅助分支按阵容留存率门控，单局 logit 绝对值不超过配置 team_aux_limit（默认 0.15）。', '',
             '赛区先验仍为六赛区归一化权重：LCK .4、LPL .3、LEC .1，其余三个各 .2/3；国际赛更新后总和保持 1。跨赛区转会保留个人相对能力，比赛环境仍采用现赛区先验。', '',
             '## 训练与边界', '',
             f'联赛预训练 {report["protocol"]["domestic_pretraining_n"]} 个分组样本，截至 2026-04-01 之前；历史世界赛微调 {report["protocol"]["worlds_finetuning_n"]} 个分组样本（2024/2025）；包含只监督首局的单局记录。2026 世界赛标签未参与梯度训练。', '',
             f'本次覆盖 {metadata["players"]} 个选手 handle，更新 {metadata["updated_games"]} 个历史单局，目标阵容缺失位置 {metadata["missing_target_player_slots"]}。历史评估假设首局实际首发在 BP 前可用；未来预测未提供阵容时使用最后已知首发。', '',
             '**个体能力无法仅凭固定五人阵容的共同胜负完全识别。个人对线与比赛统计提供额外信息，但本模型不能把相关性解释成独立个人贡献。源数据无稳定选手 ID，使用规范化 handle，重名和更名需维护映射。**', '',
             '## 同一 412 场回顾性对照', '', '| 模型 | Accuracy | Log Loss | Brier |', '|---|---:|---:|---:|',
             '| 旧默认校准 Transformer | 69.17% | 0.593892 | 0.203021 |']
    for name, value in results.items():
        s = value['reused_412']
        lines.append(f'| {name} | {s["accuracy"]:.2%} | {s["log_loss"]:.6f} | {s["brier"]:.6f} |')
    verification_path = OUT/'verification.json'
    if verification_path.exists():
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        lines += ['', '## 产物核验', '',
                  f'每个种子模型 {verification["parameters_per_model"]:,} 个参数，三个种子均已从保存权重恢复。截断至 2026 年前重建 {verification["prefix_samples"]:,} 个样本，所有选手、历史、阵容磨合及环境特征逐值相同；恢复概率最大误差 {verification["max_restored_series_probability_error"]:.8f}。', '',
                  '另已核验队名变化不影响选手核心、换人不继承旧人能力、新组织可预测、组织分支上限、交换双方概率互补、历史日期与配置混用拦截。详情见 verification.json 与 transfer_example.json。']
    lines += ['', 'player_only_ablation 使用同一已训练网络、同一校准参数，仅将组织辅助输出置零；不在评估集上重新拟合。', '',
              '**该集合和 IG–LYON 赛果已经查看，所有结果均为回顾性比较，不能当成新的独立验证。**', '',
              '## 使用', '', '```powershell',
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py train --model roster_transformer --max-pretrain-epochs 12 --max-finetune-epochs 8",
              "& 'C:\\Users\\ASUS\\.conda\\envs\\cxz\\python.exe' main.py test 'Invictus Gaming' LYON --model roster_transformer --date 2026-10-28 --patch 16.20 --best-of 3", '```', '',
              '用 --roster-a / --roster-b 提供 top,jng,mid,bot,sup 的 JSON；新组织也可预测，需提供五人阵容，必要时用 --region-a / --region-b 指定赛区。', '',
              '按用户指定的建模单位将默认入口设为 roster_transformer；这是架构选择，不表示回测优于旧模型。可用 --model calibrated_transformer 调用旧模型。', '',
              '旧模型与产物保留。新产物位于 artifacts_roster/，模型实现 models/roster_models.py，选手状态 data_gen/roster_core.py。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
