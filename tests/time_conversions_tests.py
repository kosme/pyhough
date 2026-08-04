from time_conversions import *
import numpy as np
import unittest
import random

DEVELOPMENT = False


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_gps2mjd(unittest.TestCase):
    def test_input_types(self):
        # Basic types that must be rejected
        self.assertRaises(ValueError, gps2mjd, '0')
        self.assertRaises(ValueError, gps2mjd, 0)
        self.assertRaises(ValueError, gps2mjd, True)
        self.assertRaises(ValueError, gps2mjd, ['a', 'b'])
        self.assertRaises(ValueError, tdt2tdb, object())
        self.assertRaises(ValueError, tdt2tdb, None)

        # Check expected input types do not raise an exception
        try:
            gps2mjd(0.0)  # float
            gps2mjd([0.0])  # list
            gps2mjd((0.0, 1.0))  # tuple
            gps2mjd(np.asarray(0.0))  # numpy array-like
        except ValueError:
            self.fail()

    def test_output_type(self):
        self.assertIsInstance(gps2mjd(0.0), np.ndarray,
                              "Failure to produce correct output type from a float value")
        self.assertIsInstance(gps2mjd([0.0]), np.ndarray,
                              "Failure to produce correct output type from list")
        self.assertIsInstance(gps2mjd((0.0, 1.0)), np.ndarray,
                              "Failure to produce correct output type from tuple")
        self.assertIsInstance(gps2mjd(np.asarray(0.0)), np.ndarray,
                              "Failure to produce correct output type from an array-like value")

    def test_output_value(self):
        # MJD at GPS epoch (6-Jan-1980 00:00:00)
        self.assertAlmostEqual(gps2mjd(0.0), 44244.0)
        # MJD at GPS epoch (8-Jul-2026 04:48:36)
        self.assertAlmostEqual(gps2mjd(1467521334.0), 61229.20041667)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_leapseconds(unittest.TestCase):
    MIN_VALUE = 41317.0
    MAX_VALUE = 57754.0
    RANGE = MAX_VALUE - MIN_VALUE

    def test_input_types(self):
        # Basic types that must be rejected
        self.assertRaises(TypeError, leap_seconds, '0')
        self.assertRaises(TypeError, leap_seconds, 41317)
        self.assertRaises(TypeError, leap_seconds, True)
        self.assertRaises(TypeError, leap_seconds, object())
        self.assertRaises(TypeError, leap_seconds, None)
        self.assertRaises(ValueError, leap_seconds, ['a', 'b'])

        # Check expected input types do not raise an exception
        try:
            leap_seconds(self.MIN_VALUE)  # float
            leap_seconds([self.MIN_VALUE])  # list
            leap_seconds((self.MIN_VALUE, self.MIN_VALUE))  # tuple
            leap_seconds(np.asarray(self.MIN_VALUE))  # numpy array-like
        except ValueError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(leap_seconds(self.MIN_VALUE),
                              (int, float, np.ndarray))
        self.assertIsInstance(leap_seconds([self.MIN_VALUE]),
                              (int, float, np.ndarray))
        self.assertIsInstance(leap_seconds((self.MIN_VALUE, self.MIN_VALUE)),
                              (int, float, np.ndarray))
        self.assertIsInstance(leap_seconds(np.asarray(self.MIN_VALUE)),
                              (int, float, np.ndarray))

    def test_input_values(self):
        # Random value below the first defined value
        self.assertRaises(ValueError, leap_seconds,
                          random.random() * self.MIN_VALUE)

        # Check that values in range do not throw an exception
        try:
            # Generate a random value inside the defined range
            randVal = (random.random() * self.RANGE) + self.MIN_VALUE
            leap_seconds(randVal)
        except ValueError:
            self.fail()

        # Check that values betyond the range do not throw an exception
        try:
            # Generate a random value inside the defined range
            randVal = (random.random() * self.RANGE) + self.MAX_VALUE + 1
            leap_seconds(randVal)
        except ValueError:
            self.fail()

    def test_output_values(self):
        self.assertEqual(leap_seconds(self.MIN_VALUE), 10)
        self.assertListEqual(
            list(leap_seconds([self.MIN_VALUE, self.MAX_VALUE])), [10.0, 37.0])

        # Generate a random value inside the defined range
        randVal = (random.random() * self.RANGE) + self.MIN_VALUE
        res = leap_seconds(randVal)
        self.assertGreaterEqual(res, 10.0)
        self.assertLessEqual(res, 37.0)

        # Generate a random value greater than the max defined value
        randVal = (random.random() * self.RANGE) + self.MAX_VALUE + 1
        self.assertEqual(leap_seconds(randVal), 37.0)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_tdt2tdb(unittest.TestCase):
    def test_input_types(self):
        self.assertRaises(ValueError, tdt2tdb, 'a')
        self.assertRaises(ValueError, tdt2tdb, True)
        self.assertRaises(ValueError, tdt2tdb, [12])
        self.assertRaises(ValueError, tdt2tdb, (12, 13))
        self.assertRaises(ValueError, tdt2tdb, object())
        self.assertRaises(ValueError, tdt2tdb, None)

        # Check expected input types do not raise an exception
        try:
            tdt2tdb(123)
            tdt2tdb(123.0)
            tdt2tdb(np.asarray([123], dtype=float))
        except ValueError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(tdt2tdb(123), float)
        self.assertIsInstance(tdt2tdb(123.0), float)
        self.assertIsInstance(tdt2tdb(np.asarray([123.0, 123.0])), np.ndarray)

    def test_input_values(self):
        # Check that the input supports both negative and positive values
        try:
            tdt2tdb(0)
            tdt2tdb(475.0)
            tdt2tdb(-10.0)
        except ValueError:
            self.fail()

    def test_output_values(self):
        self.assertAlmostEqual(tdt2tdb(0), -0.001181, 6)
        self.assertAlmostEqual(tdt2tdb(475.0), 0.001494, 6)
        self.assertAlmostEqual(tdt2tdb(-10.0), -0.001365, 6)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_gmst(unittest.TestCase):
    def test_input_types(self):
        self.assertRaises(ValueError, gmst, 'a')
        self.assertRaises(ValueError, gmst, True)
        self.assertRaises(ValueError, gmst, 1)
        self.assertRaises(ValueError, gmst, (12.0, 13.0))
        self.assertRaises(ValueError, gmst, [12.0])
        self.assertRaises(ValueError, gmst, object())
        self.assertRaises(ValueError, gmst, None)

        # Check expected input types do not raise an exception
        try:
            gmst(123.0)
            gmst(np.asarray([123], dtype=float))
        except ValueError:
            self.fail()

    def test_output_types(self):
        self.assertIsInstance(gmst(12.0), float)
        self.assertIsInstance(gmst(np.asarray([12.0, 13.0])), np.ndarray)

    def test_output_values(self):
        SCALE = 100
        lst = []
        # Check positive input values
        for i in range(10):
            a = random.random() * SCALE
            lst.append(a)
            st = gmst(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        # Check negative input values
        for i in range(10):
            a = random.random() * SCALE * -1
            lst.append(a)
            st = gmst(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        # Check big values
        for i in range(10):
            a = random.random() * SCALE * SCALE * SCALE
            lst.append(a)
            st = gmst(a)
            self.assertGreaterEqual(st, 0)
            self.assertLessEqual(st, 24)

        st = gmst(np.asarray(lst))
        if np.any(st < 0) or np.any(st > 24):
            self.fail()


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_mjuliandate(unittest.TestCase):
    def setUp(self):
        self.func = mjuliandate

    def tearDown(self):
        self.func = None

    def test_input_number(self):
        self.assertRaises(TypeError, self.func)
        self.assertRaises(TypeError, self.func, 1, 2)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4, 5)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4, 5, 6, 7)

        self.assertRaises(ValueError, self.func, [])
        self.assertRaises(ValueError, self.func, [1])
        self.assertRaises(ValueError, self.func, [1, 2])
        self.assertRaises(ValueError, self.func, [1, 2, 3, 4])
        self.assertRaises(ValueError, self.func, [1, 2, 3, 4, 5])
        self.assertRaises(ValueError, self.func, [1, 2, 3, 4, 5, 6, 7])

        try:
            self.func([1, 2, 3])
            self.func((1, 2, 3))
            self.func(1, 2, 3)
            self.func(1, 2, 3, 4, 5, 6)
        except TypeError:
            self.fail()

    def test_input_types(self):
        self.assertRaises(ValueError, self.func, ['a', object(), 'None'])
        self.assertRaises(ValueError, self.func, [[1, 2, 3], [1, 2]])

        try:
            self.func(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            self.func(1, 2, 3)
            self.func([[1, 2, 3], [4.0, 5.0, 6.0]])
            self.func([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3, 4, 5, 6]])
        except ValueError:
            self.fail()

    def test_input_values_date_year(self):
        # Bad values
        self.assertRaises(ValueError, self.func,
                          random.random(), 2, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func,
                          random.randint(-10000, -1), 2, 3, 4, 5, 6)
        # Good values
        try:
            self.func(1, 2, 3, 4, 5, 6)
            self.func(random.randint(2, 10000), 2, 3, 4, 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_date_month(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 13, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 0, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, -1, 3, 4, 5, 6)
        # Good values
        try:
            self.func(1, 1, 3, 4, 5, 6)
            self.func(1, 12, 3, 4, 5, 6)
            self.func(1, random.randint(2, 11), 3, 4, 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_date_day(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 1, 31 +
                          random.randint(1, 31), 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 1, 0, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 1,
                          random.randint(-31, -1), 4, 5, 6)
        # Good values
        try:
            self.func(1, 2, 1, 4, 5, 6)
            self.func(1, 2, 31, 4, 5, 6)
            self.func(1, 2, random.randint(2, 30), 4, 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_time_hour(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, -1, 5, 6)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 24, 5, 6)
        # Good values
        try:
            self.func(1, 2, 3, 0, 5, 6)
            self.func(1, 2, 3, 23, 5, 6)
            self.func(1, 2, 3, random.randint(2, 22), 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_time_minute(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, -1, 6)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 60, 6)
        # Good values
        try:
            self.func(1, 2, 3, 4, 0, 6)
            self.func(1, 2, 3, 4, 59, 6)
            self.func(1, 2, 3, 4, random.randint(2, 58), 6)
        except ValueError:
            self.fail()

    def test_input_values_time_second(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 5, -1)
        self.assertRaises(ValueError, self.func, 1, 2, 3, 4, 5, 60)
        # Good values
        try:
            self.func(1, 2, 3, 4, 5, 0)
            self.func(1, 2, 3, 4, 5, 59)
            self.func(1, 2, 3, 4, 5, 59 + random.random())
            self.func(1, 2, 3, 4, 5, random.randint(2, 58))
        except ValueError:
            self.fail()

    def test_output_values(self):
        self.assertAlmostEqual(self.func(2001, 2, 3, 4, 5, 0), 51943.17013889)
        self.assertAlmostEqual(self.func(1980, 1, 1, 0, 0, 0), 44239.00000000)


@unittest.skipIf(DEVELOPMENT, "Development")
class Test_utc2gps(unittest.TestCase):
    def setUp(self):
        self.func = utc2gps

    def tearDown(self):
        self.func = None

    def test_input_arg_number(self):
        # Test rejection of wrong numbeer of arguments
        self.assertRaises(TypeError, self.func)
        self.assertRaises(TypeError, self.func, 1, 2)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4, 5)
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4, 5, 6, 7)

        try:
            self.func("2023-05-24T22:00:00")
            self.func(1, 2, 3)
            self.func(1, 2, 3, 4, 5, 6)
        except ValueError:
            # Ignore errors related to arg values
            pass
        except TypeError:
            # Fail on arg type exceptions
            self.fail()

    def test_input_correct_number_bad_type(self):
        # One arg, wrong type (must be a string)
        self.assertRaises(TypeError, self.func, 1)
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(TypeError, self.func, True)
        self.assertRaises(TypeError, self.func, [1])

        # Three args, wrong types (must all be numbers)
        self.assertRaises(TypeError, self.func, '1', object(), None)
        self.assertRaises(TypeError, self.func, True, (True), [None])
        self.assertRaises(TypeError, self.func, 1, (True), 3)

        # Six args, wrong types (must all be numbers)
        self.assertRaises(TypeError, self.func, '1',
                          object(), None, True, (True), [1])
        self.assertRaises(TypeError, self.func, 1, 2, 3, 4, 5, [1])

    def test_input_bad_ISO_values(self):
        # Bad string
        self.assertRaises(ValueError, self.func, "bad string")
        # Almost correct input (Extra characters)
        self.assertRaises(ValueError, self.func, "2023-05-24T22:00:00UTC+8")
        # Missing field from date
        self.assertRaises(ValueError, self.func, "2023-05")

    def test_input_good_ISO_values(self):
        try:
            # Should not fail if missing time field(s) from ISO string
            self.func("2023-05-24")
            self.func("2023-05-24T22")
            self.func("2023-05-24T23:00")
            # Should not fail with a good ISO string
            self.func("2023-05-24T22:00:00")
        except ValueError:
            self.fail()

    def test_input_values_date_year(self):
        # Bad values
        self.assertRaises(ValueError, self.func,
                          random.random(), 2, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func,
                          random.randint(-10000, -1), 2, 3, 4, 5, 6)
        # Good values
        try:
            self.func(1972, 2, 3, 4, 5, 6)
            self.func(random.randint(1973, 10000), 2, 3, 4, 5, 6)
        except ValueError as ex:
            self.fail()

    def test_input_values_date_month(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 13, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 0, 3, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, -1, 3, 4, 5, 6)
        # Good values
        try:
            self.func(1972, 1, 3, 4, 5, 6)
            self.func(1972, 12, 3, 4, 5, 6)
            self.func(1972, random.randint(2, 11), 3, 4, 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_date_day(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 1, 31 +
                          random.randint(1, 31), 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 1, 0, 4, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 1,
                          random.randint(-31, -1), 4, 5, 6)
        # Good values
        try:
            self.func(1972, 2, 1, 4, 5, 6)
            self.func(1972, 2, 31, 4, 5, 6)
            self.func(1972, 2, random.randint(2, 30), 4, 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_time_hour(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, -1, 5, 6)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 24, 5, 6)
        # Good values
        try:
            self.func(1972, 2, 3, 0, 5, 6)
            self.func(1972, 2, 3, 23, 5, 6)
            self.func(1972, 2, 3, random.randint(2, 22), 5, 6)
        except ValueError:
            self.fail()

    def test_input_values_time_minute(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, -1, 6)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 60, 6)
        # Good values
        try:
            self.func(1972, 2, 3, 4, 0, 6)
            self.func(1972, 2, 3, 4, 59, 6)
            self.func(1972, 2, 3, 4, random.randint(2, 58), 6)
        except ValueError:
            self.fail()

    def test_input_values_time_second(self):
        # Bad values
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 5, -1)
        self.assertRaises(ValueError, self.func, 1972, 2, 3, 4, 5, 60)
        # Good values
        try:
            self.func(1972, 2, 3, 4, 5, 0)
            self.func(1972, 2, 3, 4, 5, 59)
            self.func(1972, 2, 3, 4, 5, 59 + random.random())
            self.func(1972, 2, 3, 4, 5, random.randint(2, 58))
        except ValueError:
            self.fail()

    def test_output_type(self):
        self.assertEqual(type(self.func(2001, 2, 3, 4, 5, 0)), float)

    def test_output_values(self):
        self.assertAlmostEqual(
            self.func(2001, 2, 3, 4, 5, 0), 665208313.000, 6)
        self.assertAlmostEqual(self.func(1980, 1, 1, 0, 0, 0), -432000.000, 6)


class Test_mjd2gps(unittest.TestCase):
    def setUp(self):
        self.func = mjd2gps

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        self.assertRaises(TypeError, self.func, 0)
        self.assertRaises(TypeError, self.func, 'a')
        self.assertRaises(TypeError, self.func, ['a'])
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, True)

        # Check the expected types don't fail
        try:
            self.func(44250.0)
            self.func(np.asarray([41319.0, 54321.123]))
        except TypeError:
            self.fail()

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
