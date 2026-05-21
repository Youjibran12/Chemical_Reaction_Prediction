def levenshtein_accuracy(pred, actual):
    m, n = len(pred), len(actual)
    if m == 0 and n == 0: return 1.0
    if m == 0 or  n == 0: return 0.0
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == actual[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return 1.0 - dp[m][n] / max(m, n)

def char_accuracy(pred, actual):
    if not actual: return 0.0
    return sum(p == a for p, a in zip(pred, actual)) / len(actual)

def prefix_match(pred, actual):
    if not actual: return 0.0
    count = 0
    for p, a in zip(pred, actual):
        if p == a: count += 1
        else: break
    return count / len(actual)

def bleu(pred, actual):
    smoother = SmoothingFunction().method1
    return sentence_bleu([list(actual)], list(pred), weights=(0.5, 0.5),
                         smoothing_function=smoother)

def top_k_accuracy(preds_lists, actuals, k):

    hits = sum(1 for cands, ref in zip(preds_lists, actuals) if ref in cands[:k])
    return hits / len(actuals) if actuals else 0.0

def compute_metrics(preds, actuals, beam_cands=None):
    char_accs     = [char_accuracy(p, a)       for p, a in zip(preds, actuals)]
    prefix_accs   = [prefix_match(p, a)        for p, a in zip(preds, actuals)]
    lev_accs      = [levenshtein_accuracy(p, a) for p, a in zip(preds, actuals)]
    bleu_scores   = [bleu(p, a)                for p, a in zip(preds, actuals)]
    exact_matches = [1.0 if p == a else 0.0    for p, a in zip(preds, actuals)]
    # Top-K Accuracy using full beam candidates when available
    if beam_cands is None:
        beam_cands = [[p] for p in preds]   # fallback: single candidate per sample
    topk_accs = {k: top_k_accuracy(beam_cands, actuals, k) for k in [1, 2, 3, 5]}
    y_true, y_pred = [], []
    for p, a in zip(preds, actuals):
        max_l = max(len(p), len(a))
        y_pred.extend(list(p.ljust(max_l)))
        y_true.extend(list(a.ljust(max_l)))
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return dict(
        char_accuracy        = np.mean(char_accs),
        prefix_match         = np.mean(prefix_accs),
        levenshtein_accuracy = np.mean(lev_accs),
        bleu                 = np.mean(bleu_scores),
        f1                   = f1,
        exact_match          = np.mean(exact_matches),
        levenshtein_scores   = lev_accs,
        bleu_scores          = bleu_scores,
        top_k_accuracy       = topk_accs,
    )

def print_metrics(name, metrics):
    print(f"\n{'='*52}\n  Results — {name}\n{'='*52}")
    print(f"{'Character-level Accuracy':<32} {100*metrics['char_accuracy']:>9.2f}%")
    print(f"{'Prefix Match':<32} {100*metrics['prefix_match']:>9.2f}%")
    print(f"{'Levenshtein Accuracy':<32} {100*metrics['levenshtein_accuracy']:>9.2f}%")
    print(f"{'BLEU Score (bigram)':<32} {metrics['bleu']:>10.4f}")
    print(f"{'Micro F1 Score':<32} {100*metrics['f1']:>9.2f}%")
    print(f"{'Exact Match Accuracy':<32} {100*metrics['exact_match']:>9.2f}%")
    for k in [1, 2, 3, 5]:
        val = 100 * metrics['top_k_accuracy'][k]
        print(f"{'Top-' + str(k) + ' Accuracy':<32} {val:>9.2f}%")
    print('='*52)