from time_conversions import *
import numpy as np
import unittest

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
        self.assertRaises(ValueError, leap_seconds, '0')
        self.assertRaises(ValueError, leap_seconds, 41317)
        self.assertRaises(ValueError, leap_seconds, True)
        self.assertRaises(ValueError, leap_seconds, ['a', 'b'])
        self.assertRaises(ValueError, leap_seconds, object())
        self.assertRaises(ValueError, leap_seconds, None)

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
        import random
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
        import random
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


if __name__ == '__main__':
    unittest.main()
