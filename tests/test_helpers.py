import unittest


class Test_Helpers(unittest.TestCase):
    def assert_raises_exception(self, expected_exception, *args):
        """
        Iterate over a series of argument cases and assert that they raise the expected Exception
        """
        assert not expected_exception is None and issubclass(
            expected_exception, Exception), "exception must have an Exception type"
        for arg in args:
            self.assertRaises(expected_exception, self.func, *arg)

    def assert_not_raises_exception(self, expected_exception, *args, ignore_exception=None):
        """
        Iterate over a series of argument cases (lists of arguments) and assert that no Exception of the expected type is raised.
        Other (optional) kinds of Exceptions can be ignored if required.
        """
        assert not expected_exception is None and issubclass(
            expected_exception, Exception), "exception must have an Exception type"
        assert not expected_exception is ignore_exception, "expected_exception cannot be ignored"
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

    def assert_raises_exception_over_arg_size_range(self, expected_exception, sample_arg, *args):
        """
        Iterate over a value range, generating variable length inputs, and asserting an exception is raised
        """
        if len(args) == 1:
            minVal = 0
            maxVal = args[0]
            step = 1
        elif len(args) == 2:
            minVal = args[0]
            maxVal = args[1]
            step = 1
        elif len(args) == 3:
            minVal = args[0]
            maxVal = args[1]
            step = args[2]
        else:
            raise ValueError(
                "One, two, or three arguments must be provided for the range.\n(endVal), (minVal, endVal), or (minVal, endVal, step)")

        if minVal < 0:
            raise ValueError(
                "Start value for the range must be equal or greater than 0.")
        if not type(minVal) == int:
            raise TypeError("Minimum value for the range must be an integer.")
        if not type(maxVal) == int:
            raise TypeError("Maximum value for the range must be an integer.")
        if not type(step) == int:
            raise TypeError("Stepping value for the range must be an integer.")
        if not maxVal > minVal:
            raise ValueError(f"Maximum value must be greater than {minVal}.")
        if not step > 0:
            raise ValueError("Step value must be greater than zero.")

        # Initialize to the starting length
        generated_args = [sample_arg]*minVal

        for x in range(minVal, maxVal, step):
            self.assertRaises(expected_exception, self.func, *generated_args)
            generated_args = generated_args + [sample_arg]*step
        self.assertRaises(expected_exception, self.func, *generated_args)
