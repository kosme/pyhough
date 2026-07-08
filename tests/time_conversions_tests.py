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


if __name__ == '__main__':
    unittest.main()
