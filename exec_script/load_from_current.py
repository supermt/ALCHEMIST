import sys
import os

# coding=utf8
import re

MODULE_NAME = "DOTA_SF"
regex = r".*%s\/" % MODULE_NAME

test_str = __file__
matches = re.search(regex, test_str)
module_path = matches.group(0)
sys.path.insert(0,module_path)

# end_dir = os.getcwd().split("/")[-1]
# print(os.getcwd().split("/"))
# if "exec_script" == end_dir:
#     # execute inside the exec directories
#     sys.path.insert(0, "../")
# elif MODULE_NAME == end_dir:
#     sys.path.insert(0, "./")
