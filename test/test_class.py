from oticscream import Icscream
import openturns as ot
import pytest
import openturns.testing as ott


@pytest.fixture(scope="session")
def data():
    """Provide some data"""
    return ot.Normal().getSample(10)


def test_class(data):
    value = 4.0
    ott.assert_almost_equal(2.0 * 2.0, value)
