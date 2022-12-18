import os
import re

folder_path = r"E:\Dropbox (BGU DAL BBB Group)\Nir_Epilepsy_Controls\Epilepsy\Results r_3"

# List all files and folders in the folder
items = os.listdir(folder_path)

# Use a regular expression to find all 4-digit numbers in the names of the items
numbers = []
for item in items:
    matches = re.findall(r'\d{4}', item)
    numbers.extend(matches)

print(numbers)

list_1 = ['1866', '4648', '5372', '8378', '5397', '5862', '5942', '5694', '9307', '8054', '0804', '2898', '2899', '7084', '0671', '2211', '9876', '1474', '4108', '4511', '9719', '5953', '3254', '0265', '2476', '1437', '4933', '5115', '2767', '7023', '9785', '1959', '3007', '6538', '7206']

set_1 = set(list_1)
set_2 = set(numbers)

# Numbers present in either set_1 or set_2, but not both
symmetric_difference = set_1.symmetric_difference(set_2)
print(symmetric_difference)
