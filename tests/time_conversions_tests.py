from time_conversions import *
import numpy as np
import unittest
import test_helpers
import random

DEVELOPMENT = False


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_gps2mjd(unittest.TestCase):
    def setUp(self):
        self.func = gps2mjd

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        # Basic types that must be rejected
        self.assertRaises(TypeError, self.func, '0')
        self.assertRaises(TypeError, self.func, 0)
        self.assertRaises(TypeError, self.func, True)
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(ValueError, self.func, ['a', 'b'])

        # Check expected input types do not raise an exception
        try:
            self.func(0.0)  # float
            self.func([0.0])  # list
            self.func((0.0, 1.0))  # tuple
            self.func(np.asarray(0.0))  # numpy array-like
        except TypeError:
            self.fail()

    def test_output_type(self):
        self.assertIsInstance(self.func(0.0), np.ndarray,
                              "Failure to produce correct output type from a float value")
        self.assertIsInstance(self.func([0.0]), np.ndarray,
                              "Failure to produce correct output type from list")
        self.assertIsInstance(self.func((0.0, 1.0)), np.ndarray,
                              "Failure to produce correct output type from tuple")
        self.assertIsInstance(self.func(np.asarray(0.0)), np.ndarray,
                              "Failure to produce correct output type from an array-like value")

    def test_output_value(self):
        # MJD at GPS epoch (6-Jan-1980 00:00:00)
        self.assertAlmostEqual(self.func(0.0), 44244.0)
        # MJD at GPS epoch (8-Jul-2026 04:48:36)
        self.assertAlmostEqual(self.func(1467521334.0), 61229.20041667)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_leapseconds(unittest.TestCase):
    MIN_VALUE = 41317.0
    MAX_VALUE = 57754.0
    RANGE = MAX_VALUE - MIN_VALUE

    def setUp(self):
        self.func = leap_seconds

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        # Basic types that must be rejected
        self.assertRaises(TypeError, self.func, '0')
        self.assertRaises(TypeError, self.func, 41317)
        self.assertRaises(TypeError, self.func, True)
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(ValueError, self.func, ['a', 'b'])

        # Check expected input types do not raise an exception
        try:
            self.func(self.MIN_VALUE)  # float
            self.func([self.MIN_VALUE])  # list
            self.func((self.MIN_VALUE, self.MIN_VALUE))  # tuple
            self.func(np.asarray(self.MIN_VALUE))  # numpy array-like
        except TypeError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(self.func(self.MIN_VALUE), float)
        self.assertIsInstance(self.func([self.MIN_VALUE]), np.ndarray)
        self.assertIsInstance(self.func(
            (self.MIN_VALUE, self.MIN_VALUE)), np.ndarray)
        self.assertIsInstance(self.func(np.asarray(self.MIN_VALUE)), float)

    def test_input_values(self):
        # Random value below the first defined value
        self.assertRaises(ValueError, self.func,
                          random.random() * self.MIN_VALUE)

        # Check that values in range do not throw an exception
        try:
            # Generate a random value inside the defined range
            randVal = (random.random() * self.RANGE) + self.MIN_VALUE
            self.func(randVal)
        except ValueError:
            self.fail()

        # Check that values beyond the range do not throw an exception
        try:
            # Generate a random value inside the defined range
            randVal = (random.random() * self.RANGE) + self.MAX_VALUE + 1
            self.func(randVal)
        except ValueError:
            self.fail()

    def test_output_values(self):
        self.assertEqual(self.func(self.MIN_VALUE), 10)
        self.assertListEqual(
            list(self.func([self.MIN_VALUE, self.MAX_VALUE])), [10.0, 37.0])

        # Generate a random value inside the defined range
        randVal = (random.random() * self.RANGE) + self.MIN_VALUE
        res = self.func(randVal)
        self.assertGreaterEqual(res, 10.0)
        self.assertLessEqual(res, 37.0)

        # Generate a random value greater than the max defined value
        randVal = (random.random() * self.RANGE) + self.MAX_VALUE + 1
        self.assertEqual(self.func(randVal), 37.0)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_tdt2tdb(unittest.TestCase):
    def setUp(self):
        self.func = tdt2tdb

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        self.assertRaises(TypeError, self.func, 'a')
        self.assertRaises(TypeError, self.func, True)
        self.assertRaises(TypeError, self.func, [12])
        self.assertRaises(TypeError, self.func, (12, 13))
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)

        # Check expected input types do not raise an exception
        try:
            self.func(123)
            self.func(123.0)
            self.func(np.asarray([123], dtype=float))
        except TypeError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(self.func(123), float)
        self.assertIsInstance(self.func(123.0), float)
        self.assertIsInstance(
            self.func(np.asarray([123.0, 123.0])), np.ndarray)

    def test_input_values(self):
        # Check that the input supports both negative and positive values
        try:
            self.func(0)
            self.func(475.0)
            self.func(-10.0)
        except ValueError:
            self.fail()

    def test_output_values(self):
        self.assertAlmostEqual(self.func(0), -0.001181, 6)
        self.assertAlmostEqual(self.func(475.0), 0.001494, 6)
        self.assertAlmostEqual(self.func(-10.0), -0.001365, 6)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_gmst(unittest.TestCase):
    def setUp(self):
        self.func = gmst

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        self.assertRaises(ValueError, self.func, 'a')
        self.assertRaises(ValueError, self.func, True)
        self.assertRaises(ValueError, self.func, 1)
        self.assertRaises(ValueError, self.func, (12.0, 13.0))
        self.assertRaises(ValueError, self.func, [12.0])
        self.assertRaises(ValueError, self.func, object())
        self.assertRaises(ValueError, self.func, None)

        # Check expected input types do not raise an exception
        try:
            self.func(123.0)
            self.func(np.asarray([123], dtype=float))
        except ValueError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(self.func(12.0), float)
        self.assertIsInstance(self.func(np.asarray([12.0, 13.0])), np.ndarray)

    def test_output_values(self):
        SCALE = 100
        lst = []
        # Check positive input values
        for i in range(10):
            a = random.random() * SCALE
            lst.append(a)
            st = self.func(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        # Check negative input values
        for i in range(10):
            a = random.random() * SCALE * -1
            lst.append(a)
            st = self.func(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        # Check big values
        for i in range(10):
            a = random.random() * SCALE * SCALE * SCALE
            lst.append(a)
            st = self.func(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        # Check mixed range values in an ndarray
        st = self.func(np.asarray(lst))
        if np.any(st < 0) or np.any(st > 24):
            self.fail()


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_mjuliandate(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = mjuliandate

    def tearDown(self):
        self.func = None

    def test_input_number(self):
        # No args, 2 args, 4 args, 5 args, 7 args
        self.assert_raises_exception(TypeError, [], [1]*2,
                                     [1]*4, [1]*5, [1]*7)

        # Empty list, 1-elem list, 2-elem list, 4-elem list, 5-elem list, 7-elem list
        self.assert_raises_exception(ValueError, [[]], [[1]],
                                     [[1]*2], [[1]*4], [[1]*5], [[1]*7])

        # 3 args, 6 args, 3-elem tuple, 6-elem tuple, 3-elem list, 6-elem list,
        self.assert_not_raises_exception(TypeError, [1]*3, [1]*6, [(1, 2, 3)],
                                         [(1, 2, 3, 4, 5, 6)], [[1]*3], [[1]*6])

    def test_input_types(self):
        # 1 arg, wrong type
        self.assertRaises(IndexError, self.func, 1)

        # 3 args, wrong types
        self.assertRaises(ValueError, self.func, ['a', object(), 'None'])

        # 1 arg, list with different sized elements
        self.assertRaises(ValueError, self.func, [[1]*3, [1]*2])

        # 6 floats, 3 ints, list with size 3 elements, list with size 6 elements
        self.assert_not_raises_exception(ValueError, [1.0]*6, [1]*3,
                                         [[[1]*3, [2.0]*3]], [[[1.0]*6, [1]*6]])

    def test_input_values_date_year(self):
        # Bad values
        self.assertRaises(ValueError, self.func,
                          random.random(), 2, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func,
                          random.randint(-10000, -1), 2, 3, 4, 5, 6)
        # Good values
        test_vals = []
        for y in [1, random.randint(2, 10000), random.randint(2, 10000)]:
            test_vals.append([y, 2, 3, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_date_month(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 13, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 0, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, -1, 3, 4, 5, 6)
        # Good values
        test_vals = []
        for m in [1, 12, random.randint(2, 11), random.randint(2, 11)]:
            test_vals.append([1, m, 3, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_date_day(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 1, 31 +
                          random.randint(1, 31), 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 1, 0, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 1,
                          random.randint(-31, -1), 4, 5, 6)
        # Good values
        test_vals = []
        for d in [1, 31, random.randint(2, 30), random.randint(2, 30)]:
            test_vals.append([1, 3, d, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_hour(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, -1, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 24, 5, 6)
        # Good values
        test_vals = []
        for h in [0, 23, random.randint(1, 22), random.randint(1, 22)]:
            test_vals.append([1, 2, 3, h, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_minute(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, -1, 6)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 60, 6)
        # Good values
        test_vals = []
        for m in [0, 59, random.randint(1, 58), random.randint(1, 58)]:
            test_vals.append([1, 2, 3, 4, m, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_second(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 5, -1)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 5, 60)
        # Good values
        test_vals = []
        for s in [0, 59, random.randint(1, 58), random.randint(1, 58),
                  59 + random.random(), 59 + random.random()]:
            test_vals.append([1, 2, 3, 4, 5, s])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_output_values(self):
        self.assertAlmostEqual(self.func(2001, 2, 3, 4, 5, 0), 51943.17013889)
        self.assertAlmostEqual(self.func(1980, 1, 1, 0, 0, 0), 44239.00000000)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_utc2gps(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = utc2gps

    def tearDown(self):
        self.func = None

    def test_input_arg_string_number(self):
        # Test rejection of wrong number of arguments
        self.assert_raises_exception(TypeError,
                                     [], ["1"]*2, ["1"]*3, ["1"]*4, ["1"]*5, ["1"]*6)

        # Test that correct number of arguments are accepted
        self.assert_not_raises_exception(TypeError,
                                         ["2023-05-24T22:00:00"],
                                         ["1"],
                                         ["a"],
                                         ignore_exception=ValueError)

    def test_input_arg_int_number(self):
        # Test rejection of wrong number of arguments
        self.assert_raises_exception(TypeError,
                                     [], [1], [1]*2, [1]*4, [1]*5, [1]*7)

        # Test that correct number of arguments are accepted
        self.assert_not_raises_exception(TypeError,
                                         [1]*3, [1]*6, ignore_exception=ValueError)

    def test_input_one_arg_bad_type(self):
        # One arg, wrong type (must be a string)
        self.assert_raises_exception(TypeError,
                                     [1], [object()], [None], [True], [[1]])

    def test_input_three_args_bad_types(self):
        # Three args, wrong types (all must be numbers)}
        self.assert_raises_exception(TypeError,
                                     [1, object(), None],
                                     [True, (True), '1'],
                                     [1.0, True, 3])

    def test_input_six_args_bad_types(self):
        # Six args, wrong types (must all be numbers)
        self.assert_raises_exception(TypeError,
                                     ['1', object(), None, True,
                                      (True), [1]],
                                     [1, 2, 3, 4, 5, [1]]
                                     )

    def test_input_bad_ISO_values(self):
        # Bad string
        self.assertRaises(ValueError, self.func, "bad string")
        # Almost correct input (Extra characters)
        self.assertRaises(ValueError, self.func, "2023-05-24T22:00:00UTC+8")
        # Missing field from date
        self.assertRaises(ValueError, self.func, "2023-05")

    def test_input_good_ISO_values(self):
        # Should not fail if missing minutes or seconds time fields from ISO string
        self.assert_not_raises_exception(ValueError,
                                         ["2023-05-24"],
                                         ["2023-05-24T22"],
                                         ["2023-05-24T23:00"])
        # Should not fail with a good, completely filled ISO string
        self.assert_not_raises_exception(ValueError,
                                         ["2023-05-24T22:00:00"])

    def test_input_values_date_year(self):
        # Bad values
        self.assertRaises(ValueError, self.func,
                          random.random(), 2, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func,
                          random.randint(-10000, -1), 2, 3, 4, 5, 6)
        # Good values
        test_vals = []
        for y in [1972, random.randint(1973, 10000), random.randint(1973, 10000)]:
            test_vals.append([y, 2, 3, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_date_month(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 13, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 0, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, -1, 3, 4, 5, 6)
        # Good values
        test_vals = []
        for m in [1, 12, random.randint(2, 11), random.randint(2, 11)]:
            test_vals.append([1972, m, 3, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_date_day(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 1, 31 +
                          random.randint(1, 31), 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 1, 0, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 1,
                          random.randint(-31, -1), 4, 5, 6)
        # Good values
        test_vals = []
        for d in [1, 31, random.randint(2, 30), random.randint(2, 30)]:
            test_vals.append([1972, 3, d, 4, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_hour(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, -1, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 24, 5, 6)
        # Good values
        test_vals = []
        for h in [0, 23, random.randint(1, 22), random.randint(1, 22)]:
            test_vals.append([1972, 2, 3, h, 5, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_minute(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, -1, 6)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 60, 6)
        # Good values
        test_vals = []
        for m in [0, 59, random.randint(1, 58), random.randint(1, 58)]:
            test_vals.append([1972, 2, 3, 4, m, 6])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_input_values_time_second(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 5, -1)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 5, 60)
        # Good values
        test_vals = []
        for s in [0, 59, random.randint(1, 58), random.randint(1, 58),
                  59 + random.random(), 59 + random.random()]:
            test_vals.append([1972, 2, 3, 4, 5, s])
        self.assert_not_raises_exception(ValueError, *test_vals)

    def test_output_type(self):
        self.assertEqual(type(self.func(2001, 2, 3, 4, 5, 0)), float)

    def test_output_values(self):
        self.assertAlmostEqual(
            self.func(2001, 2, 3, 4, 5, 0), 665208313.000, 6)
        self.assertAlmostEqual(self.func(1980, 1, 1, 0, 0, 0), -432000.000, 6)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_mjd2gps(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = mjd2gps

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        self.assert_raises_exception(TypeError,
                                     [0], ['a'], [['a']], [None], [object()], [True])

        # Check the expected types don't fail
        self.assert_not_raises_exception(TypeError,
                                         [44250.0], [np.asarray([41319.0, 54321.123])])

    def test_output_type(self):
        self.assertIsInstance(self.func(44250.0), float)
        self.assertIsInstance(
            self.func(np.asarray([41319.0, 54321.123])), np.ndarray)

    def test_output_value(self):
        self.assertAlmostEqual(self.func(44250.0), 518400.000)
        for val, expected in zip(self.func(np.asarray([41319.0, 54321.123])), [-252720009.000, 870663441.200]):
            self.assertAlmostEqual(val, expected, 6)


if __name__ == '__main__':
    unittest.main()
