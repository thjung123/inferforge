import pytest
from gateway.utils.exceptions import InvalidInputError, TritonConnectionError


def test_invalid_input_error():
    with pytest.raises(InvalidInputError) as e:
        raise InvalidInputError("Bad input")
    assert e.value.status_code == 400


def test_triton_connection_error():
    with pytest.raises(TritonConnectionError) as e:
        raise TritonConnectionError()
    assert e.value.status_code == 503
