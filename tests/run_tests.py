import unittest

loader = unittest.TestLoader()
tests = loader.discover('./cases/', pattern='*.py')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
