"""Sample original textual values for column type discovery."""

from collections import deque
from random import Random

import pandas as pd


def character_pattern(value: str) -> str:
    """Replace letters and digits while preserving separators and other characters."""
    pattern = []
    for char in value:
        if char.isalpha():
            pattern.append('A')
        elif char.isdigit():
            pattern.append('9')
        else:
            pattern.append(char)
    return ''.join(pattern)


def sample_character_patterns(
    column: pd.Series,
    budget: int,
    seed: int,
    inverted_index: dict[str, set[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Return prompt samples and the subset reserved as known-type representatives."""
    if budget < 0:
        raise ValueError('The CTD sample budget must be non-negative.')
    if budget == 0:
        return [], []

    values = sorted(value for value in column.dropna().astype(str).unique() if value.strip())
    if len(values) <= budget:
        return values, []

    rng = Random(seed)
    index = inverted_index or {}
    values_by_type = {}
    for value in values:
        for label in index.get(value, ()):
            values_by_type.setdefault(label, []).append(value)

    # Leave at least half the budget for pattern sampling when novel values exist.
    reserve = budget // 2 if any(not index.get(value) for value in values) else budget
    representatives = set()
    covered_types = set()
    known_types = sorted(values_by_type)
    rng.shuffle(known_types)
    for label in known_types:
        if len(representatives) == reserve:
            break
        if label in covered_types:
            continue
        value = rng.choice(values_by_type[label])
        representatives.add(value)
        covered_types.update(index[value])

    groups = {}
    for value in values:
        groups.setdefault(character_pattern(value), []).append(value)
    ranked_patterns = sorted(groups, key=lambda pattern: (-len(groups[pattern]), pattern))
    represented_patterns = {character_pattern(value) for value in representatives}
    for pattern in groups:
        groups[pattern] = [value for value in groups[pattern] if value not in representatives]

    remaining = budget - len(representatives)
    unseen_patterns = [pattern for pattern in ranked_patterns if pattern not in represented_patterns]
    if len(unseen_patterns) > remaining:
        common_slots = (remaining + 1) // 2
        order = unseen_patterns[:common_slots] + rng.sample(unseen_patterns[common_slots:], remaining - common_slots)
    else:
        order = unseen_patterns + [pattern for pattern in ranked_patterns if pattern in represented_patterns]

    active = deque(pattern for pattern in order if groups[pattern])
    typed_samples = sorted(representatives)
    samples = typed_samples.copy()
    # Cycle through nonempty groups, so any unused allocation goes to the other groups.
    while active and len(samples) < budget:
        pattern = active.popleft()
        group = groups[pattern]
        index_in_group = rng.randrange(len(group))
        group[index_in_group], group[-1] = group[-1], group[index_in_group]
        samples.append(group.pop())
        if group:
            active.append(pattern)

    return samples, typed_samples
