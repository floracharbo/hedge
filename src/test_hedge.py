from src.hedge import HEDGE

generator = HEDGE(
    n_homes=10,
    factors_gen='matrix',
    n_consecutive_days=2,
    brackets_definition='percentile'
)

for i in range(10):
    day = generator.make_next_day(plotting=True)
