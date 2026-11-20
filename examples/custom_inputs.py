"""Run from any directory after installing metric-private-matching."""
import numpy as np
from metric_private_matching import MECHANISMS, match


def main():
    generator = np.random.default_rng(2026)
    private = generator.normal(size=(30, 5))
    public = generator.normal(size=(30, 5))
    for mechanism in MECHANISMS:
        permutation = match(private, public, 4., mechanism=mechanism, rng=71)
        # True cost is an offline diagnostic; it is not a private release.
        cost = np.linalg.norm(private-public[permutation], axis=1).sum()
        print(f'{mechanism}: first five partners={permutation[:5]}, total cost={cost:.3f}')


if __name__ == '__main__':
    main()
