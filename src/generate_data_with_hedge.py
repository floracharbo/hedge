from src.hedge import HEDGE


def generate_data_with_hedge(plotting=True):
    generator = HEDGE(
        n_homes=10,
        n_consecutive_days=2,
        brackets_definition='percentile',
    )

    for _ in range(10):
        day = generator.make_next_day(plotting=plotting)

if __name__ == "__main__":
    generate_data_with_hedge()
