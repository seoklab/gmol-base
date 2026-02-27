from gmol.base.data.mmcif import Assembly


def test_save_load(sample_assembly: Assembly):
    j1 = sample_assembly.model_dump_json(indent=2)
    data = Assembly.model_validate_json(j1)

    j2 = data.model_dump_json(indent=2)

    assert j1 == j2
