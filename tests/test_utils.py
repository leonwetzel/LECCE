import unittest

from lecce.utilities.tools import is_directory_empty


class TestUtilsMethods(unittest.TestCase):

    def test_is_directory_empty(self):
        """Tests if we can check if a given directory is empty or not.

        Returns
        -------

        """
        self.assertFalse(is_directory_empty(f"{ROOT_DIR}"), False)


if __name__ == '__main__':
    unittest.main()
