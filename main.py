"""Unified entry point: `train` fits the model chosen in config.toml, `test` forecasts with it.

config.toml 的 model 决定走哪条流水线：elo_mlp（Elo + NumPy MLP）或 dense / dense_transformer
（PyTorch MatchNet），以及表格模型、概率校准和专家融合。命令行参数可临时覆盖配置值。
"""
import argparse
from collections import deque
import csv
from datetime import datetime
import html
import json
import math
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import random
import time
import tomllib
import numpy as np
import torch
from torch.nn import functional as F
from core import (CONTEXT_NAMES, DIFF_NAMES, LEAGUES, ROOT, TeamState, bootstrap, fit_temperature, load_series, logit,
                  make_features, metrics, series_probability, sigmoid, team_key)
from data_gen.features_v2 import History, ROLES, history_sequence, player_vector
from models.deep_models import MODEL_NAMES as DEEP_MODELS, build_model, series_logits
from models.elo_mlp import ResidualMLP
from models.tabular_models import NAMES as TABULAR_MODELS

OUT = ROOT/'artifacts_deep'
MODEL_NAMES = ('elo_mlp', *DEEP_MODELS, 'hist_gbm', 'expert_blend', 'calibrated_transformer', 'regional_transformer', 'roster_transformer', 'roster_transformer_2026', 'roster_transformer_2026_swiss', 'roster_transformer_2026_swiss_weighted')
DEFAULTS = {'model': 'roster_transformer_2026_swiss_weighted', 'seed': 42, 'max_pretrain_epochs': 60,
            'max_finetune_epochs': 40, 'stage': 'finetuned', 'seeds': [42, 43, 44]}


def load_config(path):
    config = dict(DEFAULTS)
    file = Path(path)
    if file.exists():
        config.update(tomllib.loads(file.read_text(encoding='utf-8')))
    if config['model'] not in MODEL_NAMES:
        raise SystemExit(f'config model must be one of {list(MODEL_NAMES)}, got {config["model"]!r}')
    return config


# --------------------------------------------------------------------------- elo_mlp pipeline

def arrays(samples):
    return (np.stack([s['d'] for s in samples]), np.stack([s['c'] for s in samples]),
            np.array([s['elo_logit'] for s in samples]), np.array([s['y'] for s in samples], dtype=float))


def fit(train, val, l2, seed=42):
    d, c, offset, y = train
    vd, vc, vo, vy = val
    model = ResidualMLP(d.shape[1], c.shape[1], seed)
    m = {k: np.zeros_like(v) for k, v in model.params.items()}
    v = {k: np.zeros_like(w) for k, w in model.params.items()}
    best_loss, best_epoch, best_params = float('inf'), 0, None
    rng, step, trace = np.random.default_rng(seed), 0, []
    for epoch in range(301):
        val_loss = metrics(vy, sigmoid(model.logits(vd, vc, vo)))['log_loss']
        trace.append({'epoch': epoch, 'validation_log_loss': val_loss})
        if val_loss < best_loss-1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_params = {k: w.copy() for k, w in model.params.items()}
        if epoch-best_epoch >= 35 or epoch == 300:
            break
        order = rng.permutation(len(y))
        for batch in range(0, len(y), 128):
            idx = order[batch:batch+128]
            grads = model.gradients(d[idx], c[idx], offset[idx], y[idx], l2)
            step += 1
            for key, grad in grads.items():
                grad = np.clip(grad, -2, 2)
                m[key] = .9*m[key]+.1*grad
                v[key] = .999*v[key]+.001*grad*grad
                model.params[key] -= .002*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
    model.params = best_params
    return model, {'l2': l2, 'seed': seed, 'best_epoch': best_epoch, 'validation_log_loss': best_loss, 'trace': trace}


def split_samples(samples):
    # Boundaries selected from date coverage only, before training or inspecting scores.
    dates = sorted({s['date'] for s in samples})
    boundaries = [dates[int(len(dates)*q)] for q in (.60, .75, .85)]
    groups = [[], [], [], []]
    purged = []
    for s in samples:
        split = sum(s['date'] >= b for b in boundaries)
        if split < 3 and s['end_date'] >= boundaries[split]:
            purged.append(s['id'])
        else:
            groups[split].append(s)
    if min(map(len, groups)) < 30:
        raise ValueError(f'Not enough independent series for four temporal partitions: {list(map(len, groups))}')
    return groups, boundaries, purged


def write_html(report, predictions, path):
    rows = ''.join('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{values[k]:.4f}</td>' for k in ['log_loss', 'brier', 'accuracy', 'ece_10_bins'])+'</tr>' for name, values in report['test'].items())
    def calibration_svg(values):
        circles = ''.join(f'<circle cx="{50+b["prediction"]*260:.1f}" cy="{310-b["observed"]*260:.1f}" r="5"><title>n={b["n"]}; predicted={b["prediction"]:.3f}; observed={b["observed"]:.3f}</title></circle>' for b in values['calibration_bins'])
        return f'<svg viewBox="0 0 360 360" role="img" aria-label="Calibration plot"><path d="M50 50V310H310 M50 310L310 50" fill="none" stroke="#b6becd"/>{circles}<text x="110" y="345">Predicted probability</text><text x="10" y="28">Observed win frequency</text></svg>'
    splits = ''.join(f'<li>{name}: {v["n"]} series, {v["first"]} to {v["last"]}</li>' for name, v in report['splits'].items())
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LoL 赛前预测回测</title>
<style>body{{font:16px/1.7 system-ui;background:#f4f6fb;color:#17233b;max-width:1050px;margin:40px auto;padding:24px}}h1{{font-size:30px}}section{{background:white;padding:24px;margin:20px 0;border-radius:14px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px;border-bottom:1px solid #eee}}.plots{{display:flex;gap:24px}}svg{{max-width:330px;fill:#287ce0}}code{{word-break:break-word}}</style>
<h1>英雄联盟 · BP 前系列赛胜率回测</h1><p>实际历史数据；按系列赛开始日期切分；严格冻结历史特征。此报告不代表当前比赛的预测。</p>
<section><h2>数据与时间切分</h2><ul>{splits}</ul><p>数据来源：ChainCC 开放数据集，CC BY 4.0。<a href="https://chaincc.lol/free/data">数据主页</a>。详见 <code>data/sources.json</code> 的 URL、日期覆盖及 SHA-256。</p></section>
<section><h2>未参与训练、选型及校准的测试集</h2><table><tr><th>模型</th><th>Log Loss ↓</th><th>Brier ↓</th><th>Accuracy ↑</th><th>ECE ↓</th></tr>{rows}</table>
<p>MLP − 校准 Elo 的 Log Loss 差异：{report['comparison']['delta_log_loss_mlp_minus_elo']:.4f}；描述性 bootstrap 95% 区间：{report['comparison']['paired_series_bootstrap_95_ci']}。负数代表 MLP 较好。</p></section>
<section><h2>校准曲线</h2><div class="plots"><div><h3>校准 Elo</h3>{calibration_svg(report['test']['elo_calibrated'])}</div><div><h3>Elo + MLP（校准）</h3>{calibration_svg(report['test']['mlp_calibrated'])}</div></div><p>对角线表示预测概率与实际频率一致；点的样本数可悬停查看。小样本区间可能不稳定。</p></section>
<section><h2>模型与边界</h2><ul><li>所有比赛日的预测只使用前一日及更早已完成的系列赛；同系列赛内不更新预测特征。</li><li>12 个双方差异特征，19 个共同环境特征；两层 tanh，16/8 隐藏单元；反对称残差显式保证交换互补。</li><li>标准化仅在训练集拟合；验证集选正则和早停；独立校准集拟合温度；测试集仅报告。</li><li>BO3/BO5 根据小局序列与终场比分回溯重建，并检查数据源提供的非空赛制；未与官方赛程核验。单局及歧义系列赛排除，快照最后一天排除。数据缺局仍可能导致赛制误标。</li><li>版本假定在预测时已知，来自首局记录；仅使用版本编号和历史版本经验，尚未学习英雄改动的机制，也未接入赛前阵容公告、官方赛程与历史发布时间。</li><li>跨赛区仅通过国际比赛更新共享 Elo，没有声称已解决赛区标尺；队伍以稳定 ID 跟踪，尚无选手、转会和更名映射模型。</li><li>系列赛 Elo 基线采用独立同胜率小局假设；未建模 Fearless Draft、未来选边、选手位置或 BP。</li><li>测试期间模型权重与温度固定，Elo 和历史状态随已完成比赛滚动更新；这是一次时间留出回测，不是多折 walk-forward。</li></ul></section></html>'''
    path.write_text(text, encoding='utf-8')


def train_elo_mlp(config):
    series, audit = load_series(ROOT/'data'/'teams.csv')
    print('DATA AUDIT', json.dumps(audit), flush=True)
    samples, states = make_features(series)
    groups, boundaries, purged = split_samples(samples)
    names = ['train', 'validation', 'calibration', 'test']
    split_meta = {name: {'n': len(group), 'first': group[0]['date'], 'last': group[-1]['date'],
                         'bo3': sum(s['best_of'] == 3 for s in group), 'bo5': sum(s['best_of'] == 5 for s in group)}
                  for name, group in zip(names, groups)}
    print('SPLITS', json.dumps(split_meta), flush=True)
    sets = [arrays(g) for g in groups]
    # Zero-centred RMS preserves d -> -d exactly. Context remains shared.
    dscale = np.maximum(np.sqrt(np.mean(sets[0][0]**2, axis=0)), .05)
    cmean, cscale = sets[0][1].mean(axis=0), np.maximum(sets[0][1].std(axis=0), .1)
    sets = [(np.clip(d/dscale, -6, 6), (c-cmean)/cscale, o, y) for d, c, o, y in sets]
    trials = []
    for l2 in (.001, .01, .1):
        model, trial = fit(sets[0], sets[1], l2, config['seed'])
        print('TRIAL', json.dumps({k: v for k, v in trial.items() if k != 'trace'}), flush=True)
        trials.append((model, trial))
    model, selected = min(trials, key=lambda item: item[1]['validation_log_loss'])
    cd, cc, co, cy = sets[2]
    elo_temp = fit_temperature(co, cy)
    mlp_temp = fit_temperature(model.logits(cd, cc, co), cy)
    td, tc, to, ty = sets[3]
    mlp_logits = model.logits(td, tc, to)
    probabilities = {'coin_flip': np.full(len(ty), .5), 'elo': sigmoid(to),
                     'elo_calibrated': sigmoid(to/elo_temp), 'mlp': sigmoid(mlp_logits),
                     'mlp_calibrated': sigmoid(mlp_logits/mlp_temp)}
    complement_error = float(np.max(np.abs(sigmoid(mlp_logits/mlp_temp)+sigmoid(model.logits(-td, tc, -to)/mlp_temp)-1)))
    report = {'audit': audit, 'split_policy': '60/15/10/15 percent of distinct series-start dates; no random split',
              'boundaries': boundaries, 'purged_boundary_series': purged, 'splits': split_meta,
              'selected_trial': {k: v for k, v in selected.items() if k != 'trace'},
              'elo_temperature': elo_temp, 'mlp_temperature': mlp_temp,
              'test': {name: metrics(ty, p) for name, p in probabilities.items()},
              'comparison': bootstrap(ty, probabilities['mlp_calibrated'], probabilities['elo_calibrated']),
              'swap_max_error': complement_error,
              'by_best_of': {str(bo): {name: metrics(ty[np.array([s['best_of'] == bo for s in groups[3]])], p[np.array([s['best_of'] == bo for s in groups[3]])])
                                      for name, p in probabilities.items()} for bo in [3, 5] if any(s['best_of'] == bo for s in groups[3])},
              'features': {'difference': DIFF_NAMES, 'context': CONTEXT_NAMES},
              'parameter_count': sum(w.size for w in model.params.values())}
    out = ROOT/'artifacts'
    out.mkdir(exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (out/'training_history.json').write_text(json.dumps([t for _, t in trials], indent=2), encoding='utf-8')
    np.savez_compressed(out/'model.npz', **model.params, dscale=dscale, cmean=cmean, cscale=cscale,
                        elo_temperature=elo_temp, mlp_temperature=mlp_temp)
    feature_rows = []
    for name, group in zip(names, groups):
        for s in group:
            row = {k: v for k, v in s.items() if k not in ('d', 'c')}
            row.update(zip(['diff_'+f for f in DIFF_NAMES], s['d']))
            row.update(zip(['context_'+f for f in CONTEXT_NAMES], s['c']))
            row['split'] = name
            feature_rows.append(row)
    with (out/'series_features.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    predictions = []
    for i, sample in enumerate(groups[3]):
        row = {k: sample[k] for k in ['id', 'date', 'league', 'team_a', 'team_b', 'best_of', 'y']}
        row.update({name: float(p[i]) for name, p in probabilities.items()})
        predictions.append(row)
    with (out/'test_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    state_json = {key: {'rating': s.rating, 'history': list(s.history), 'last_day': s.last_day,
                        'last_year': s.last_year, 'roster': sorted(s.roster)} for key, s in states.items()}
    (out/'team_states.json').write_text(json.dumps(state_json, indent=2), encoding='utf-8')
    aliases = {r['teamname']: team_key(r) for s in series for game in s.games for r in game}
    (out/'inference.json').write_text(json.dumps({'aliases': aliases, 'snapshot_last_day': max(s.end for s in series),
                                                'train_last_day': groups[0][-1]['date'],
                                                'validation_last_day': groups[1][-1]['date'],
                                                'calibration_last_day': groups[2][-1]['date']}, indent=2), encoding='utf-8')
    write_html(report, predictions, out/'report.html')
    print('RESULT', json.dumps({k: {m: v for m, v in value.items() if m != 'calibration_bins'} for k, value in report['test'].items()}), flush=True)
    print('COMPARISON', json.dumps(report['comparison']), flush=True)
    print('SWAP_ERROR', complement_error, flush=True)
    print('Saved artifacts to', out, flush=True)


def test_elo_mlp(args, config, parser):
    info = json.loads((ROOT/'artifacts'/'inference.json').read_text(encoding='utf-8'))
    if args.date <= info['snapshot_last_day']:
        parser.error('The saved state includes this date or later games; use test_predictions.csv for historical predictions.')
    lag = (datetime.fromisoformat(args.date)-datetime.fromisoformat(info['snapshot_last_day'])).days
    raw_states = json.loads((ROOT/'artifacts'/'team_states.json').read_text(encoding='utf-8'))
    def resolve(name):
        options = {k.casefold(): v for k, v in info['aliases'].items()}
        key = name if name in raw_states else options.get(name.casefold())
        if key not in raw_states:
            parser.error(f'Unknown team {name!r}; use a full name from artifacts/inference.json')
        raw = raw_states[key]
        state = TeamState(raw['rating'], deque(raw['history'], maxlen=15), raw['last_day'], set(raw['roster']), raw['last_year'])
        return key, state.features(args.date, set(), args.patch)
    a, xa = resolve(args.team_a)
    b, xb = resolve(args.team_b)
    if a == b:
        parser.error('Choose two different teams.')
    d = xa-xb
    minor = float(args.patch.split('.')[1])
    major = float(args.patch.split('.')[0])
    c = np.array([major/20, minor/30, args.best_of/5, (xa[9]+xb[9])/2]+[float(args.league == league) for league in LEAGUES])
    base = float(logit(series_probability(float(sigmoid(math.log(10)*d[0])), args.best_of)))
    archive = np.load(ROOT/'artifacts'/'model.npz', allow_pickle=False)
    model = ResidualMLP(len(d), len(c))
    model.params = {k: archive[k] for k in model.params}
    d = np.clip(d/archive['dscale'], -6, 6)[None, :]
    c = ((c-archive['cmean'])/archive['cscale'])[None, :]
    p = float(sigmoid(model.logits(d, c, np.array([base]))/archive['mlp_temperature'])[0])
    print(json.dumps({'team_a': args.team_a, 'team_b': args.team_b, 'p_a_series_win_mlp': p,
                      'p_b_series_win_mlp': 1-p,
                      'p_a_series_win_elo': float(sigmoid(base/archive['elo_temperature'])),
                      'best_of': args.best_of, 'league': args.league, 'date': args.date, 'patch': args.patch,
                      'snapshot_last_day': info['snapshot_last_day'], 'snapshot_age_days': lag,
                      'limitations': 'No current roster, draft, side or official schedule feed; see report.html.'}, indent=2))


# --------------------------------------------------------------------------- matchnet pipeline

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Normalizer:
    def fit(self, raw, ids, samples):
        d, c = raw['d'][ids], raw['c'][ids]
        self.dscale = np.maximum(np.sqrt(np.mean(d*d, axis=0)), .05)
        self.cmean, self.cscale = c.mean(axis=0), np.maximum(c.std(axis=0), .1)
        history = np.concatenate([raw['seq_a'][ids][raw['mask_a'][ids]], raw['seq_b'][ids][raw['mask_b'][ids]]])
        self.smean, self.sscale = history.mean(axis=0), np.maximum(history.std(axis=0), .1)
        leagues = sorted({samples[i]['league'] for i in ids})
        self.vocabulary = {name: i+1 for i, name in enumerate(leagues)}
        self.league_scales = {}
        for name in leagues:
            rows = np.array([i for i in ids if samples[i]['league'] == name])
            if len(rows) >= 50:
                local = np.maximum(np.sqrt(np.mean(raw['d'][rows]**2, axis=0)), .05)
                # Preserve Elo's cross-region scale; normalize style statistics within league.
                local[:7] = self.dscale[:7]
                self.league_scales[name] = local
        return self

    def transform(self, raw, samples, device):
        scales = np.stack([self.league_scales.get(s['league'], self.dscale) for s in samples])
        arrays = {k: np.array(raw[k]) for k in (raw.files if hasattr(raw, 'files') else raw.keys()) if k not in {'d', 'c', 'seq_a', 'seq_b'}}
        arrays['d'] = np.clip(raw['d']/scales, -7, 7)
        arrays['c'] = np.clip((raw['c']-self.cmean)/self.cscale, -7, 7)
        for side in ['a', 'b']:
            value = np.clip((raw['seq_'+side]-self.smean)/self.sscale, -7, 7)
            value[~raw['mask_'+side]] = 0
            arrays['seq_'+side] = value
        arrays['league'] = np.array([self.vocabulary.get(s['league'], 0) for s in samples], dtype=np.int64)
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}

    def state(self):
        return {'dscale': self.dscale.tolist(), 'cmean': self.cmean.tolist(), 'cscale': self.cscale.tolist(),
                'smean': self.smean.tolist(), 'sscale': self.sscale.tolist(), 'vocabulary': self.vocabulary,
                'league_scales': {k: v.tolist() for k, v in self.league_scales.items()}}

    @classmethod
    def from_state(cls, state):
        instance = cls()
        for key in ['dscale', 'cmean', 'cscale', 'smean', 'sscale']:
            setattr(instance, key, np.array(state[key], dtype=np.float32))
        instance.vocabulary = state['vocabulary']
        instance.league_scales = {k: np.array(v, dtype=np.float32) for k, v in state['league_scales'].items()}
        return instance


def forward(model, tensors, idx):
    return model(*[tensors[k][idx] for k in ['d', 'c', 'seq_a', 'seq_b', 'mask_a', 'mask_b', 'league', 'offset']])


@torch.no_grad()
def predict(model, tensors, ids, batch_size=512):
    model.eval()
    output = []
    for start in range(0, len(ids), batch_size):
        idx = torch.as_tensor(ids[start:start+batch_size], device=tensors['d'].device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tensors['d'].is_cuda):
            output.append(forward(model, tensors, idx).float().cpu().numpy())
    return np.concatenate(output)


def load_model(checkpoint, name, device):
    model = build_model(name, checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def joint_loss(game, tensors, idx):
    counts = tensors['game_count'][idx].float()
    map_loss = (F.binary_cross_entropy_with_logits(game.float(), tensors['map_target'][idx].float(), reduction='none')*counts).sum()/counts.sum()
    bo = tensors['best_of'][idx]
    valid = bo >= 3
    series_loss = F.binary_cross_entropy_with_logits(series_logits(game[valid], bo[valid]), tensors['y'][idx][valid].float()) if valid.any() else map_loss
    return .5*map_loss+.5*series_loss


def val_summary(model, tensors, ids):
    logits = predict(model, tensors, ids)
    idx = torch.as_tensor(ids, device=tensors['d'].device)
    game = torch.as_tensor(logits, device=idx.device)
    score = float(joint_loss(game, tensors, idx).cpu())
    return score


def fit_stage(model, tensors, train_ids, val_ids, stage, max_epochs, patience, fixed_epochs=None):
    fine = stage == 'worlds_finetune'
    if fine:
        head = [p for name, p in model.named_parameters() if name.startswith('head.')]
        body = [p for name, p in model.named_parameters() if not name.startswith('head.')]
        optimizer = torch.optim.AdamW([{'params': body, 'lr': 2e-5}, {'params': head, 'lr': 1e-4}], weight_decay=.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    device = tensors['d'].device
    ids = torch.as_tensor(train_ids, device=device)
    best, best_epoch, best_state, trace = float('inf'), 0, None, []
    total_epochs = fixed_epochs or max_epochs
    start_time = time.monotonic()
    for epoch in range(1, total_epochs+1):
        model.train()
        order = ids[torch.randperm(len(ids), device=device)]
        losses = []
        for start in range(0, len(ids), 128 if fine else 256):
            batch = order[start:start+(128 if fine else 256)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                game = forward(model, tensors, batch)
            loss = joint_loss(game, tensors, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss in {stage}, epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = val_summary(model, tensors, val_ids) if val_ids is not None else float(np.mean(losses))
        trace.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_joint_loss': validation if val_ids is not None else None})
        if fixed_epochs is None and validation < best-1e-5:
            best, best_epoch = validation, epoch
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == total_epochs:
            print(json.dumps({'stage': stage, **trace[-1], 'elapsed_s': round(time.monotonic()-start_time, 1)}), flush=True)
        if fixed_epochs is None and epoch-best_epoch >= patience:
            break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = total_epochs
    return {'stage': stage, 'best_epoch': best_epoch, 'best_validation_joint_loss': best if val_ids is not None else None,
            'epochs_run': epoch, 'seconds': time.monotonic()-start_time, 'trace': trace}


def raw_series(logits, bo):
    return series_logits(torch.from_numpy(logits), torch.from_numpy(bo)).numpy().astype(float)


def train_matchnet(config):
    name, seed = config['model'], config['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('high')
    raw = np.load(OUT/'data.npz', allow_pickle=False)
    samples = json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    domestic = np.array([s['domain'] == 'domestic' for s in samples])
    years = np.array([s['worlds_year'] for s in samples])
    dates = np.array([s['date'] for s in samples])
    ends = np.array([s['end_date'] for s in samples])
    train_select = np.flatnonzero(domestic & (ends < '2025-07-01'))
    val_select = np.flatnonzero(domestic & (dates >= '2025-07-01') & (ends < '2025-10-14'))
    worlds_train = np.flatnonzero(years == '2024')
    worlds_val = np.flatnonzero(years == '2025')
    final_train = np.flatnonzero(domestic & (ends < '2026-04-01'))
    final_fine = np.flatnonzero((years == '2024') | (years == '2025'))
    calibration = np.flatnonzero(domestic & (dates >= '2026-04-01') & (ends < '2026-05-24'))
    worlds_test = np.flatnonzero(years == '2026')
    old = {r['id']: r for r in csv.DictReader((ROOT/'artifacts'/'test_predictions.csv').open(encoding='utf-8'))}
    benchmark = np.array([i for i, s in enumerate(samples) if s['id'] in old])
    assert len(benchmark) == 412
    assert all(int(old[samples[i]['id']]['y']) == raw['y'][i] for i in benchmark)
    protocol = {'seed': seed, 'device': str(device), 'torch': str(torch.__version__),
                'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                'selection_pretrain': {'n': len(train_select), 'end_exclusive': '2025-07-01'},
                'selection_domestic_validation': {'n': len(val_select), 'start': '2025-07-01', 'end_exclusive': '2025-10-14'},
                'selection_worlds_train': {'year': 2024, 'n': len(worlds_train)},
                'selection_worlds_validation': {'year': 2025, 'n': len(worlds_val)},
                'final_domestic_train': {'n': len(final_train), 'end_exclusive': '2026-04-01'},
                'final_worlds_finetune': {'years': [2024, 2025], 'n': len(final_fine)},
                'calibration': {'n': len(calibration), 'start': '2026-04-01', 'end_exclusive': '2026-05-24'},
                'worlds_2026_evaluation': {'n_first_series_maps': len(worlds_test), 'n_bo3_bo5': int((raw['best_of'][worlds_test] >= 3).sum())},
                'reused_benchmark': {'n': len(benchmark), 'status': 'previously inspected; retrospective comparison'},
                'worlds_2026_optimizer_labels_used': 0,
                'loss': '0.5 scoreline-weighted per-map BCE + 0.5 complete-series BCE',
                'architecture': {'dense_widths': [1024, 768, 512, 256], 'transformer_layers': 4, 'd_model': 128, 'heads': 8, 'history_games_per_team': 12}}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print('PROTOCOL', json.dumps(protocol), flush=True)
    seed_everything(seed)
    folder = OUT/f'{name}_seed{seed}'
    folder.mkdir(exist_ok=True)
    normalizer = Normalizer().fit(raw, train_select, samples)
    tensors = normalizer.transform(raw, samples, device)
    config_dims = {'nd': raw['d'].shape[1], 'nc': raw['c'].shape[1], 'leagues': len(normalizer.vocabulary)+1,
                   'sequence_length': 12}
    model = build_model(name, config_dims).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print('MODEL', name, 'parameters', parameters, flush=True)
    selected_pretrain = fit_stage(model, tensors, train_select, val_select, 'domestic_selection', config['max_pretrain_epochs'], 8)
    selected_fine = fit_stage(model, tensors, worlds_train, worlds_val, 'worlds_finetune', config['max_finetune_epochs'], 8)
    # Freeze epoch choices from the forward 2025 experiment before reporting 2026.
    choices = {'pretrain_epochs': selected_pretrain['best_epoch'], 'finetune_epochs': selected_fine['best_epoch']}
    (folder/'choices.json').write_text(json.dumps(choices, indent=2), encoding='utf-8')
    del model, tensors
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    seed_everything(seed)
    normalizer = Normalizer().fit(raw, final_train, samples)
    tensors = normalizer.transform(raw, samples, device)
    config_dims['leagues'] = len(normalizer.vocabulary)+1
    model = build_model(name, config_dims).to(device)
    final_pretrain = fit_stage(model, tensors, final_train, None, 'domestic_final', choices['pretrain_epochs'], 0, fixed_epochs=choices['pretrain_epochs'])
    before = predict(model, tensors, np.arange(len(samples)))
    torch.save({'state_dict': model.state_dict(), 'config': config_dims, 'normalizer': normalizer.state(), 'stage': 'league_pretrained'}, folder/'pretrained.pt')
    final_finetune = fit_stage(model, tensors, final_fine, None, 'worlds_finetune', choices['finetune_epochs'], 0, fixed_epochs=choices['finetune_epochs'])
    after = predict(model, tensors, np.arange(len(samples)))
    cal_series = calibration[raw['best_of'][calibration] >= 3]
    results, predictions, temperatures = {}, {}, {}
    for stage, logits in [('pretrained', before), ('worlds_finetuned', after)]:
        series_raw = raw_series(logits, raw['best_of'])
        series_temp = fit_temperature(series_raw[cal_series], raw['y'][cal_series])
        map_temp = fit_temperature(logits[calibration].astype(float), raw['first_map_y'][calibration])
        series_p = sigmoid(series_raw/series_temp)
        map_p = sigmoid(logits.astype(float)/map_temp)
        world_series = worlds_test[raw['best_of'][worlds_test] >= 3]
        results[stage] = {'reused_412_series': metrics(raw['y'][benchmark], series_p[benchmark]),
                          'worlds_2026_first_map': metrics(raw['first_map_y'][worlds_test], map_p[worlds_test]),
                          'worlds_2026_series': metrics(raw['y'][world_series], series_p[world_series])}
        temperatures[stage] = {'series': series_temp, 'map': map_temp}
        predictions[stage+'_series'] = series_p
        predictions[stage+'_first_map'] = map_p
    elo_game = raw['offset'].astype(float)
    elo_series = raw_series(elo_game.astype('float32'), raw['best_of'])
    elo_series_temp = fit_temperature(elo_series[cal_series], raw['y'][cal_series])
    elo_map_temp = fit_temperature(elo_game[calibration], raw['first_map_y'][calibration])
    world_series = worlds_test[raw['best_of'][worlds_test] >= 3]
    results['elo_same_data'] = {'reused_412_series': metrics(raw['y'][benchmark], sigmoid(elo_series[benchmark]/elo_series_temp)),
                                'worlds_2026_first_map': metrics(raw['first_map_y'][worlds_test], sigmoid(elo_game[worlds_test]/elo_map_temp)),
                                'worlds_2026_series': metrics(raw['y'][world_series], sigmoid(elo_series[world_series]/elo_series_temp))}
    model.eval()
    idx = torch.as_tensor(benchmark[:32], device=device)
    with torch.no_grad():
        pos = forward(model, tensors, idx)
        neg = model(-tensors['d'][idx], tensors['c'][idx], tensors['seq_b'][idx], tensors['seq_a'][idx],
                    tensors['mask_b'][idx], tensors['mask_a'][idx], tensors['league'][idx], -tensors['offset'][idx])
        swap_error = float(torch.max(torch.abs(torch.sigmoid(pos)+torch.sigmoid(neg)-1)).cpu())
    report = {'model': name, 'parameters': parameters, 'seed': seed, 'protocol': protocol,
              'choices': choices, 'temperatures': temperatures, 'results': results, 'swap_max_error': swap_error,
              'training': [selected_pretrain, selected_fine, final_pretrain, final_finetune]}
    (folder/'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    torch.save({'state_dict': model.state_dict(), 'config': config_dims, 'normalizer': normalizer.state(),
                'stage': 'worlds_finetuned', 'temperatures': temperatures['worlds_finetuned'], 'protocol': protocol}, folder/'finetuned.pt')
    rows = []
    for i in sorted(set(benchmark.tolist()+worlds_test.tolist())):
        row = {k: samples[i][k] for k in ['id', 'date', 'league', 'domain', 'team_a', 'team_b', 'best_of', 'y', 'first_map_y']}
        row.update({k: float(v[i]) for k, v in predictions.items()})
        rows.append(row)
    with (folder/'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('RESULT', name, json.dumps({stage: {task: {k: v for k, v in values.items() if k != 'calibration_bins'} for task, values in tasks.items()} for stage, tasks in results.items()}), flush=True)
    del model, tensors
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def make_input(snapshot, a, b, day, patch, best_of, league, playoffs=0, roster_a=None, roster_b=None):
    def state(key):
        raw = snapshot['teams'][key]
        return History(ratings=np.array(raw['ratings']), series_rating=raw['series_rating'],
                       games=deque(raw['games'], maxlen=50), series=deque(raw['series'], maxlen=10),
                       last_day=raw['last_day'], year=raw['year'], roster=set(raw['roster']), roster_by_role=raw['roster_by_role'])
    sa, sb = state(a), state(b)
    xa, _ = sa.features(day, patch)
    xb, _ = sb.features(day, patch)
    h2h = next((v['outcomes'] for v in snapshot['head_to_head'] if v['a'] == a and v['b'] == b), [])
    ra = sa.roster_by_role if roster_a is None else roster_a
    rb = sb.roster_by_role if roster_b is None else roster_b
    pa, _, _ = player_vector({r: {'playername': ra.get(r) or ''} for r in ROLES}, sa.roster, snapshot['player_elo'], snapshot['player_history'])
    pb, _, _ = player_vector({r: {'playername': rb.get(r) or ''} for r in ROLES}, sb.roster, snapshot['player_elo'], snapshot['player_history'])
    d = np.r_[xa-xb, np.mean(h2h) if h2h else 0, pa-pb].astype('float32')
    if 'regional' in snapshot:
        from data_gen.regional import RegionalRatings, adjust_features
        regional = RegionalRatings.from_state(snapshot['regional'])
        d = adjust_features(regional, a, b, day, d, snapshot['player_elo'], ra, rb)
    major, minor = map(float, patch.split('.'))
    mean_old_history = (math.log1p(min(15, len(sa.games)))+math.log1p(min(15, len(sb.games))))/6
    c = np.array([major/20, minor/30, best_of/5, mean_old_history]+[float(league == l) for l in LEAGUES]+
                 [playoffs, (xa[-4]+xb[-4])/2, abs(d[2]) if 'regional' in snapshot else abs(xa[2]-xb[2]),
                  (xa[-2]+xb[-2])/2], dtype='float32')
    seq_a, mask_a = history_sequence(sa, day, patch)
    seq_b, mask_b = history_sequence(sb, day, patch)
    raw = {'d': d[None], 'c': c[None], 'seq_a': seq_a[None], 'seq_b': seq_b[None],
           'mask_a': mask_a[None], 'mask_b': mask_b[None], 'best_of': np.array([best_of]),
           'offset': np.array([math.log(10)*d[2]], dtype='float32')}
    return raw, ra, rb


def test_matchnet(args, config, parser):
    snapshot = json.loads((OUT/'snapshot.json').read_text(encoding='utf-8'))
    if args.date <= snapshot['snapshot_last_day']:
        parser.error('The saved snapshot contains this day or later results. Use historical predictions.csv for past dates.')
    aliases = {name.casefold(): key for name, key in snapshot['aliases'].items()}
    a = aliases.get(args.team_a.casefold(), args.team_a)
    b = aliases.get(args.team_b.casefold(), args.team_b)
    if a not in snapshot['teams'] or b not in snapshot['teams'] or a == b:
        parser.error('Choose two distinct teams from snapshot.json aliases.')
    def roster(value):
        if value is None:
            return None
        result = json.loads(value)
        if set(result) != set(ROLES) or not all(isinstance(v, str) for v in result.values()):
            parser.error('A supplied lineup requires all five roles with player names.')
        return result
    raw, ra, rb = make_input(snapshot, a, b, args.date, args.patch, args.best_of, args.league, args.playoffs,
                             roster(args.roster_a), roster(args.roster_b))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probabilities = []
    for seed in config['seeds']:
        folder = OUT/f'{config["model"]}_seed{seed}'
        checkpoint = torch.load(folder/f'{config["stage"]}.pt', map_location='cpu', weights_only=True)
        norm = Normalizer.from_state(checkpoint['normalizer'])
        tensors = norm.transform(raw, [{'league': args.league}], device)
        model = load_model(checkpoint, config['model'], device)
        logits = predict(model, tensors, np.array([0]))
        stage = 'worlds_finetuned' if config['stage'] == 'finetuned' else 'pretrained'
        report = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        temperature = report['temperatures'][stage]['series']
        sl = series_logits(torch.tensor(logits), torch.tensor([args.best_of]))
        probabilities.append(float(torch.sigmoid(sl/temperature)[0]))
    value = float(np.mean(probabilities))
    print(json.dumps({'team_a': args.team_a, 'team_b': args.team_b, 'p_a_win': value, 'p_b_win': 1-value,
                      'model': config['model'], 'stage': config['stage'], 'seeds': config['seeds'], 'seed_probabilities': probabilities,
                      'snapshot_last_day': snapshot['snapshot_last_day'], 'lineup_a': ra, 'lineup_b': rb,
                      'lineup_status': 'User supplied or last-known historical lineup; official current lineup is not fetched.'},
                     ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------- entry points

def train(config):
    if config['model'] == 'roster_transformer_2026_swiss_weighted':
        from pipelines.swiss_weighted import run
        run(config)
    elif config['model'] == 'roster_transformer_2026_swiss':
        from pipelines.swiss_adapt import run
        run(config)
    elif config['model'] == 'roster_transformer_2026':
        from pipelines.roster_2026 import run
        run(config)
    elif config['model'] == 'roster_transformer':
        from pipelines.roster_train import run
        run(config)
    elif config['model'] == 'regional_transformer':
        from pipelines.regional_train import run
        run(config)
    elif config['model'] == 'elo_mlp':
        train_elo_mlp(config)
    elif config['model'] in ('expert_blend', 'calibrated_transformer'):
        from pipelines.blend_experts import run
        run()
    elif config['model'] in (*TABULAR_MODELS, 'hist_gbm'):
        from pipelines.optimize import run
        run(models=[config['model']], seeds=[config['seed']])
    else:
        train_matchnet(config)


def test(args, config, parser):
    if config['model'] == 'roster_transformer_2026_swiss_weighted':
        from pipelines.swiss_weighted import test as roster_test
        roster_test(args, config, parser)
    elif config['model'] == 'roster_transformer_2026_swiss':
        from pipelines.swiss_adapt import test as roster_test
        roster_test(args, config, parser)
    elif config['model'] == 'roster_transformer_2026':
        from pipelines.roster_2026 import test as roster_test
        roster_test(args, config, parser)
    elif config['model'] == 'roster_transformer':
        from pipelines.roster_train import test as roster_test
        roster_test(args, config, parser)
    elif config['model'] == 'regional_transformer':
        from pipelines.regional_train import test as regional_test
        regional_test(args, config, parser)
    elif config['model'] == 'elo_mlp':
        test_elo_mlp(args, config, parser)
    elif config['model'] in ('expert_blend', 'calibrated_transformer'):
        from pipelines.blend_experts import test as blend_test
        blend_test(args, config, parser)
    elif config['model'] in (*TABULAR_MODELS, 'hist_gbm'):
        from pipelines.optimize import test as optimized_test
        optimized_test(args, config, parser)
    else:
        test_matchnet(args, config, parser)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'config.toml'))
    commands = parser.add_subparsers(dest='command', required=True)

    train_parser = commands.add_parser('train', help='fit the model chosen in config.toml')
    train_parser.add_argument('--model', choices=MODEL_NAMES)
    train_parser.add_argument('--seed', type=int)
    train_parser.add_argument('--max-pretrain-epochs', type=int)
    train_parser.add_argument('--max-finetune-epochs', type=int)

    optimize_parser = commands.add_parser('optimize', help='compare structured-data models with past-only rolling selection')
    optimize_parser.add_argument('--models', nargs='+', choices=[*TABULAR_MODELS, 'hist_gbm'])
    optimize_parser.add_argument('--seeds', nargs='+', type=int)
    optimize_parser.add_argument('--refresh', action='store_true')

    test_parser = commands.add_parser('test', help='forecast a future pre-BP series with the saved checkpoints')
    test_parser.add_argument('team_a')
    test_parser.add_argument('team_b')
    test_parser.add_argument('--date', required=True)
    test_parser.add_argument('--patch', required=True)
    test_parser.add_argument('--best-of', type=int, choices=[3, 5], required=True)
    test_parser.add_argument('--league', default='WCS', choices=LEAGUES)
    test_parser.add_argument('--playoffs', type=int, choices=[0, 1], default=0)
    test_parser.add_argument('--model', choices=MODEL_NAMES)
    test_parser.add_argument('--stage', choices=['pretrained', 'finetuned'])
    test_parser.add_argument('--seeds', nargs='+', type=int)
    test_parser.add_argument('--roster-a', help='JSON mapping top,jng,mid,bot,sup to announced names')
    test_parser.add_argument('--roster-b', help='JSON mapping top,jng,mid,bot,sup to announced names')
    test_parser.add_argument('--region-a', help='Region override for player-core prediction of a new organization')
    test_parser.add_argument('--region-b', help='Region override for player-core prediction of a new organization')

    args = parser.parse_args()
    config = load_config(args.config)
    overrides = {key: getattr(args, key, None) for key in ['model', 'seed', 'max_pretrain_epochs', 'max_finetune_epochs', 'stage', 'seeds']}
    config.update({key: value for key, value in overrides.items() if value is not None})
    if args.command == 'optimize':
        from pipelines.optimize import run
        run(args.models, config['seeds'], args.refresh)
        if (args.models is None or 'hist_gbm' in args.models) and (ROOT/'artifacts_optimized'/'hist_gbm_seed42'/'model.pkl').exists():
            from pipelines.blend_experts import run as blend
            blend()
            if args.models is None:
                from pipelines.report_optimized import main as write_optimized_report
                write_optimized_report()
    elif args.command == 'train':
        train(config)
    else:
        test(args, config, test_parser)


if __name__ == '__main__':
    main()
