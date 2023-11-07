from src.generate_data_with_hedge import generate_data_with_hedge


def test_generate_data_with_hedge():
    generate_data_with_hedge(plotting=False)
    assert True
