import unittest


class Test_Helpers(unittest.TestCase):
    # Iterate over a series of argument cases and assert that they raise the expected Exception
    def assert_raises_exception(self, expected_exception, *args):
        assert not expected_exception is None and issubclass(
            expected_exception, Exception), "exception must have an Exception type"
        for arg in args:
            self.assertRaises(expected_exception, self.func, *arg)

    # Iterate over a series of argument cases and assert that no Exception of the expected type is raised.
    # Other (optional) kinds of Exceptions can be ignored if required.
    def assert_not_raises_exception(self, expected_exception, *args, ignore_exception=None):
        assert not expected_exception is None and issubclass(
            expected_exception, Exception), "exception must have an Exception type"
        assert not expected_exception is ignore_exception, "exception cannot be ignored"
        if ignore_exception is None:
            try:
                for arg in args:
                    self.func(*arg)
            except expected_exception:
                self.fail()
        else:
            try:
                for arg in args:
                    self.func(*arg)
            except ignore_exception:
                pass
            except expected_exception:
                self.fail()
