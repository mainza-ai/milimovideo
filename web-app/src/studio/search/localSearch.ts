export interface SearchRecord<T> {
    item: T;
    text: string;
    kind: string;
}

export interface RankedResult<T> {
    item: T;
    kind: string;
    score: number;
}

function tokens(value: string): string[] {
    return value
        .toLowerCase()
        .split(/[^a-z0-9@]+/)
        .filter((token) => token.length > 0);
}

/**
 * Local ranked search. Not an embedding model: deterministic token overlap with
 * prefix matches and tag hits, so results are explainable and offline.
 */
export function rankSearch<T>(
    query: string,
    records: Array<SearchRecord<T>>,
    limit = 40,
): Array<RankedResult<T>> {
    const queryTokens = tokens(query);
    if (queryTokens.length === 0) return [];

    const results: Array<RankedResult<T>> = [];
    for (const record of records) {
        const recordTokens = tokens(record.text);
        let score = 0;
        for (const token of queryTokens) {
            if (recordTokens.includes(token)) score += 3;
            else if (recordTokens.some((candidate) => candidate.startsWith(token))) score += 2;
            else if (record.text.toLowerCase().includes(token)) score += 1;
        }
        if (score > 0) results.push({ item: record.item, kind: record.kind, score });
    }

    return results
        .sort((a, b) => b.score - a.score || a.kind.localeCompare(b.kind))
        .slice(0, limit);
}
