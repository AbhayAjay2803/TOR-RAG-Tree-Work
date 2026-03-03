import numpy as np

def mmr_selection(query_emb, candidates, lam, top_k):
    """Greedy MMR selection."""
    selected = []
    remaining = candidates.copy()
    while len(selected) < top_k and remaining:
        mmr_scores = []
        for doc in remaining:
            rel = doc['relevance']
            if selected:
                sim_to_selected = max(np.dot(doc['embedding'], s['embedding']) for s in selected)
            else:
                sim_to_selected = 0
            mmr = lam * rel - (1 - lam) * sim_to_selected
            mmr_scores.append(mmr)
        best_idx = np.argmax(mmr_scores)
        selected.append(remaining.pop(best_idx))
    return selected