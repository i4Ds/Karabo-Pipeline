import os

from karabo.util.data_util import get_module_absolute_path

test_path = os.path.join(get_module_absolute_path(), "test")
data_path = os.path.join(test_path, "data")
