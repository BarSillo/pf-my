import rlhedge

def test_rlhedge_load():
    # Verify that the rlhedge package has a __file__ attribute
    file_location = getattr(rlhedge, "__file__", None)
    print("rlhedge package location:", file_location)
    assert file_location is not None, "rlhedge.__file__ is not defined, package load failed"
