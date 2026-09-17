import unittest
import test_helpers
from physics import *
import numpy as np
import random


class Test_constants(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = constants

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        # Reject any type of input values
        self.assertRaises(TypeError, self.func, 0)
        self.assertRaises(TypeError, self.func, 1.0)
        self.assertRaises(TypeError, self.func, '0')
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(TypeError, self.func, True)
        # object covers tuples, lists, and dictionaries
        self.assertRaises(TypeError, self.func, object())

        # Do not raise exception when no args are provided
        self.assert_not_raises_exception(TypeError, [])

    def test_input_number(self):
        # Reject any number of inputs greater than 0.
        # Since it has been proved that the type makes no difference, only one type is checked
        self.assert_raises_exception_over_arg_size_range(TypeError, 1, 1, 6)

    def test_output_types(self):
        self.assertEqual(type(self.func()), dict)

    def test_output_keys_number(self):
        self.assertEqual(len(self.func().keys()), 21)

    def test_output_keys_values(self):
        self.assertListEqual(list(self.func().keys()),
                             ['c', 'h', 'hbar', 'hbar_inev', 'ev',
                             'Msun', 'G', 'eps0', 'fine_struct',
                              'f_earth', 'omega_earth_rot',
                              'omega_earth_orb', 'v_earth_rot',
                              'v_earth_orb', 'Rorb', 'Re', 'v0',
                              'vesc', 'rhodm', 'StellarDay', 'units']
                             )

    def test_units_keys_values(self):
        self.assertListEqual(list(self.func()['units'].keys()),
                             ['ev_to_inv_s', 'ev_to_inv_m', 'ev_to_kg', 'kg_to_ev',
                                 'charge_LH', 'charge_G', 'kpc_to_m', 'mpc_to_m']
                             )


class Test_calc_mc_with_k(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = calc_mc_with_k

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1)
        self.assertRaises(TypeError, self.func, '0')
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(TypeError, self.func, True)

        # Assert no exception is thrown for acceptable input types
        self.assert_not_raises_exception(TypeError, [1.0])
        self.assert_not_raises_exception(TypeError, [np.asarray(1)])
        self.assert_not_raises_exception(TypeError, [np.asarray([1])])

    def test_input_values(self):
        # Assert it rejects negative or zero values
        self.assertRaises(ValueError, self.func, -1e-23)
        self.assertRaises(ValueError, self.func, -1e23)
        self.assertRaises(ValueError, self.func, 0.0)
        self.assertRaises(ValueError, self.func, np.asarray([0.0]))
        self.assertRaises(ValueError, self.func, np.asarray([1, 2, 0.0, 5]))
        self.assertRaises(ValueError, self.func, np.asarray([1, 2, -1e-23, 5]))

        # Assert no exception is thrown for acceptable inputs
        self.assert_not_raises_exception(ValueError, [1e-23])
        self.assert_not_raises_exception(ValueError, [1975.17])
        self.assert_not_raises_exception(ValueError, [np.asarray(1e71)])
        self.assert_not_raises_exception(ValueError, [np.asarray([492])])

    def test_output_type(self):
        self.assertIsInstance(self.func(1.0), float)
        self.assertIsInstance(self.func(np.asarray(1)), float)
        self.assertIsInstance(self.func(np.asarray([1])), np.ndarray)

    def test_output_values(self):
        const = constants()
        for x in range(10):
            k = random.random()*100
            c_fact = (const['c']**3)/const['G']/const['Msun']
            expected = c_fact * ((5/96)*(np.pi**(-8/3))*k)**(3/5)
            self.assertAlmostEqual(self.func(k), expected)


class Test_calc_k(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = calc_k

    def tearDown(self):
        self.func = None

    def test_input_types(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1)
        self.assertRaises(TypeError, self.func, '0')
        self.assertRaises(TypeError, self.func, object())
        self.assertRaises(TypeError, self.func, None)
        self.assertRaises(TypeError, self.func, True)

        # Assert no exception is thrown for acceptable input types
        self.assert_not_raises_exception(TypeError, [1.0])
        self.assert_not_raises_exception(TypeError, [np.asarray([1e-23])])

    def test_input_values(self):
        # Assert it rejects out of range values (negative or zero values)
        self.assertRaises(ValueError, self.func, 0.0)
        self.assertRaises(ValueError, self.func, np.asarray(0))
        self.assertRaises(ValueError, self.func, -1e-23)
        self.assertRaises(ValueError, self.func, -1e23)
        self.assertRaises(ValueError, self.func, np.asarray([-1.0]))
        self.assertRaises(ValueError, self.func, np.asarray([1, 2, -1.0, 5]))

        # Assert no exception is thrown for acceptable inputs
        self.assert_not_raises_exception(ValueError, [1e-23])
        self.assert_not_raises_exception(ValueError, [1e23])
        self.assert_not_raises_exception(ValueError, [1975.17])
        self.assert_not_raises_exception(ValueError, [np.asarray(1e71)])
        self.assert_not_raises_exception(ValueError, [np.asarray([492.0])])

    def test_output_type(self):
        self.assertIsInstance(self.func(1.0), float)
        self.assertIsInstance(self.func(np.asarray(1.0)), float)
        self.assertIsInstance(self.func(np.asarray([1.0])), np.ndarray)

    # Since the output of calc_mc_with_k has been proved correct, the fact
    # that this value is its inverse corroborates the output correctness
    def test_output_values(self):
        for x in range(10):
            mc = random.random()*100
            self.assertAlmostEqual(self.func(calc_mc_with_k(mc)), mc)


class Test_fdot_chirp(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = calc_fdot_chirp

    def tearDown(self):
        self.func = None

    def test_input_types_arg1(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1, 1.0)
        self.assertRaises(TypeError, self.func, '1', 1.0)
        self.assertRaises(TypeError, self.func, object(), 1.0)
        self.assertRaises(TypeError, self.func, None, 1.0)
        self.assertRaises(TypeError, self.func, True, 1.0)

        # Assert expected types are not rejected
        self.assert_not_raises_exception(TypeError, [1.0, 1.0])
        self.assert_not_raises_exception(TypeError, [
            np.asarray([1.0, 2.3]),
            np.asarray([1.0, 2.3])
        ])

    def test_input_types_arg2(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1.0, 1)
        self.assertRaises(TypeError, self.func, 1.0, '1')
        self.assertRaises(TypeError, self.func, 1.0, object())
        self.assertRaises(TypeError, self.func, 1.0, None)
        self.assertRaises(TypeError, self.func, 1.0, True)

        # Assert expected types are not rejected
        self.assert_not_raises_exception(TypeError, [1.0, 1.0])
        self.assert_not_raises_exception(TypeError, [
            np.asarray([1.0, 2.3]),
            np.asarray([3.7, 9.2])
        ])

    def test_input_values_arg1(self):
        # Assert it rejects zero and negative values
        self.assertRaises(ValueError, self.func, 0.0, 1.0)
        self.assertRaises(ValueError, self.func, -1.0, 1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, 0.0, 1.0]),
                          np.asarray([1.0, 1.0, 1.0, 1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, -1e23, 3.17]),
                          np.asarray([1.0, 1.0, 1.0, 1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, -1e-23, 3.17]),
                          np.asarray([1.0, 1.0, 1.0, 1.0]))

        # Assert no exception is thrown for acceptable inputs
        self.assert_not_raises_exception(ValueError, [1.0, 19.57])
        self.assert_not_raises_exception(ValueError, [1e23, 19.57])
        self.assert_not_raises_exception(ValueError, [1e-23, 19.57])

    def test_input_values_arg2(self):
        # Assert it rejects out of range values (zero or negative values)
        self.assertRaises(ValueError, self.func, 1.0, -1.0)
        self.assertRaises(ValueError, self.func, 1.0, 0.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, 1.0, 1.23]),
                          np.asarray([1.0, 1.0, 0.0, 1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, 1.0, 1.23]),
                          np.asarray([1.0, 1.0, -1e-23, 1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.3, 1.0, 1.23]),
                          np.asarray([1.0, 1.0, -1e23, 1.0]))

        # Assert no exception is thrown for acceptable inputs
        self.assert_not_raises_exception(ValueError, [1.0, 19.57])
        self.assert_not_raises_exception(ValueError, [1.0, 1e23])
        self.assert_not_raises_exception(ValueError, [1.0, 1e-23])

    def test_output_type(self):
        self.assertIsInstance(self.func(1.0, 1.0), np.ndarray)
        self.assertIsInstance(self.func(np.asarray(
            [1.0]), np.asarray([1.0])), np.ndarray)
        self.assertIsInstance(self.func(np.asarray(
            [1.0, 2.0]), np.asarray([1.0, 2.0])), np.ndarray)


class Test_calc_time_to_coalescence(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = calc_time_to_coalescence

    def tearDown(self):
        self.func = None

    def test_input_arg1_bad_type(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1, 1.0)
        self.assertRaises(TypeError, self.func, '1', 1.0)
        self.assertRaises(TypeError, self.func, object(), 1.0)
        self.assertRaises(TypeError, self.func, None, 1.0)
        self.assertRaises(TypeError, self.func, True, 1.0)

    def test_input_arg2_bad_type(self):
        self.assertRaises(TypeError, self.func, 1.0, 1)
        self.assertRaises(TypeError, self.func, 1.0, '1')
        self.assertRaises(TypeError, self.func, 1.0, object())
        self.assertRaises(TypeError, self.func, 1.0, None)
        self.assertRaises(TypeError, self.func, 1.0, True)

    def test_input_good_args_type(self):
        # Assert expected types are not rejected
        self.assert_not_raises_exception(TypeError, [1.0, 1.0])
        self.assert_not_raises_exception(TypeError, [
            np.asarray([1.0, 2.3]),
            np.asarray([3.7, 9.2])
        ])

    def test_input_values(self):
        # Assert it rejects out of range values (negative or zero values)
        self.assertRaises(ValueError, self.func, 0.0, 1.0)
        self.assertRaises(ValueError, self.func, -1.0, 1.0)
        self.assertRaises(ValueError, self.func, 1.0, 0.0)
        self.assertRaises(ValueError, self.func, 1.0, -1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([-1e-23]), np.asarray([1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([-1e23]), np.asarray([1.0]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0]), np.asarray([-1e-23]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0]), np.asarray([-1e23]))
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0, 0.0, 5.0]),
                          np.asarray([1.0, 2.0, 1.0, 5.0])
                          )
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0, -1.0, 5.0]),
                          np.asarray([1.0, 2.0, 1.0, 5.0])
                          )
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0, 1.0, 5.0]),
                          np.asarray([1.0, 2.0, 0.0, 5.0])
                          )
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0, 1.0, 5.0]),
                          np.asarray([1.0, 2.0, -1.0, 5.0])
                          )

        # Assert no exception is thrown for acceptable inputs
        self.assert_not_raises_exception(ValueError, [1.0, 19.57])
        self.assert_not_raises_exception(ValueError, [1975.17, 1e-23])
        self.assert_not_raises_exception(
            ValueError, [np.asarray(1e71), np.asarray(1.23e-12)])
        self.assert_not_raises_exception(
            ValueError, [np.asarray([492]), np.asarray([492])])

    def test_output_type(self):
        self.assertIsInstance(self.func(1.0, 1.0), np.ndarray)
        self.assertIsInstance(self.func(np.asarray(
            [1.0]), np.asarray([1.0])), np.ndarray)
        self.assertIsInstance(self.func(np.asarray(
            [1.0, 2.0]), np.asarray([1.0, 2.0])), np.ndarray)


class Test_shift_x0_by_time(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = shift_x0_by_time

    def tearDown(self):
        self.func = None

    def test_input_arg1_bad_type(self):
        self.assertRaises(TypeError, self.func, 1, 1.0, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, '1.0', 1.0, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, object(), 1.0, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, None, 1.0, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, True, 1.0, 1.0, 1.0)

    def test_input_arg2_bad_type(self):
        self.assertRaises(TypeError, self.func, 1.0, 1, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, '1.0', 1.0, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, object(), 1.0, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, None, 1.0, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, True, 1.0, 1.0)

    def test_input_arg3_bad_type(self):
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, '1.0', 1.0)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, object(), 1.0)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, None, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, True, 1.0)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, np.asarray(1.0), 1.0)
        self.assertRaises(TypeError, self.func, 1.0,
                          1.0, np.asarray([1.0]), 1.0)

    def test_input_arg4_bad_type(self):
        # Assert rejects wrong input types
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, 1)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, '1.0')
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, object())
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, None)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, True)
        self.assertRaises(TypeError, self.func, 1.0, 1.0, 1.0, np.asarray(1.0))
        self.assertRaises(TypeError, self.func, 1.0,
                          1.0, 1.0, np.asarray([1.0]))

    def test_input_good_args_type(self):
        # Assert expected types are not rejected
        self.assert_not_raises_exception(TypeError, [1.0, 1.0, 1.0, 1.0])
        self.assert_not_raises_exception(TypeError, [
            np.asarray([1.0, 2.3]),
            np.asarray([3.7, 9.2]),
            1.0, 2.0
        ])
        self.assert_not_raises_exception(TypeError, [
            1.0,
            np.asarray([3.7, 9.2]),
            1.0, 2.0
        ], ignore_exception=ValueError)

    def test_input_arg2_values(self):
        self.assertRaises(ValueError, self.func, 1.0, 0.0, 0.0, 1.0)
        self.assertRaises(ValueError, self.func, 1.0, -1e23, 0.0, 1.0)
        self.assertRaises(ValueError, self.func, 1.0, -1e-23, 0.0, 1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0]),
                          np.asarray([1.0, 0.0]),
                          1.0, 1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0]),
                          np.asarray([1.0, -1e23]),
                          1.0, 1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 2.0]),
                          np.asarray([1.0, -1e-23]),
                          1.0, 1.0)

    def test_input_arg3_values(self):
        self.assertRaises(ValueError, self.func, 1.0, 1.0, 0.0, 1.0)

    def test_output_type(self):
        x0 = 1.0
        out = self.func(x0, 1.0, 1.0, 1.0)
        self.assertIsInstance(out, type(x0))

        out = self.func(x0, np.asarray([1.0]), 1.0, 1.0)
        self.assertIsInstance(out, type(x0))

        x0 = np.asarray([1.0])
        out = self.func(x0, 1.0, 1.0, 1.0)
        self.assertIsInstance(out, type(x0))

    def test_output_shape(self):
        x0 = 1.0
        out = self.func(x0, 1.0, 1.0, 1.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        out = self.func(x0, np.asarray([1.0]), 1.0, 1.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([1.0])
        out = self.func(x0, 1.0, 1.0, 1.0)
        self.assertEqual(np.shape(x0), np.shape(out))


class Test_get_f0_from_x0(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = get_f0_from_x0

    def tearDown(self):
        self.func = None

    def test_input_arg1_bad_type(self):
        self.assertRaises(TypeError, self.func, 1, 2.0)
        self.assertRaises(TypeError, self.func, '1.0', 2.0)
        self.assertRaises(TypeError, self.func, object(), 2.0)
        self.assertRaises(TypeError, self.func, None, 2.0)
        self.assertRaises(TypeError, self.func, True, 2.0)

    def test_input_arg2_bad_type(self):
        self.assertRaises(TypeError, self.func, 1.0, 1)
        self.assertRaises(TypeError, self.func, 1.0, '1.0')
        self.assertRaises(TypeError, self.func, 1.0, object())
        self.assertRaises(TypeError, self.func, 1.0, None)
        self.assertRaises(TypeError, self.func, 1.0, True)
        self.assertRaises(TypeError, self.func, 1.0, np.asarray([1.0]))

    def test_input_arg2_bad_value(self):
        self.assertRaises(ValueError, self.func, 1.0, 1.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([1.0, 1e23, 1e-23, -1e-23, -1e23]), 1.0)

    def test_output_shape(self):
        x0 = 1.0
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([1.0])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([1.0, 2.0])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([[1.0, 2.0], [1e23, 1e-23]])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))


class Test_get_x0_from_f0(test_helpers.Test_Helpers):
    def setUp(self):
        self.func = get_x0_from_f0

    def tearDown(self):
        self.func = None

    def test_input_arg1_bad_type(self):
        self.assertRaises(TypeError, self.func, 1, 2.0)
        self.assertRaises(TypeError, self.func, '1.0', 2.0)
        self.assertRaises(TypeError, self.func, object(), 2.0)
        self.assertRaises(TypeError, self.func, None, 2.0)
        self.assertRaises(TypeError, self.func, True, 2.0)

    def test_input_arg2_bad_type(self):
        self.assertRaises(TypeError, self.func, 1.0, 1)
        self.assertRaises(TypeError, self.func, 1.0, '1.0')
        self.assertRaises(TypeError, self.func, 1.0, object())
        self.assertRaises(TypeError, self.func, 1.0, None)
        self.assertRaises(TypeError, self.func, 1.0, True)
        self.assertRaises(TypeError, self.func, 1.0, np.asarray([1.0]))

    def test_input_arg1_bad_value(self):
        self.assertRaises(ValueError, self.func, 0.0, 2.0)
        self.assertRaises(ValueError, self.func, -1e-23, 2.0)
        self.assertRaises(ValueError, self.func,
                          np.asarray([0.0, 1e23, 1e-23, -1e-23, -1e23]), 3.17)

    def test_output_shape(self):
        x0 = 1.0
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([1.0])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([1.0, 2.0])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))

        x0 = np.asarray([[1.0, 2.0], [1e23, 1e-23]])
        out = self.func(x0, 2.0)
        self.assertEqual(np.shape(x0), np.shape(out))


if __name__ == '__main__':
    unittest.main()
