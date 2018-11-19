"""Process the plate barcode to plate kind mapping from a txt from a docx."""
import re
import sys
import pandas as pd

pattern = (r'(?P<useless>\(.*\) )(?P<kind>D\d{9})'
           r' = (?P<barcode>\d*-KTC-\w{3,4}).*')

if __name__ == '__main__':
    fn = sys.argv[1]
    plate_kinds = []
    plate_barcodes = []
    with open(fn) as fin:
        for line in fin:
            match = re.match(pattern, line)
            if match is not None:
                plate_kinds.append(match['kind'])
                plate_barcodes.append(match['barcode'])

    data = pd.DataFrame({'plate_kind': plate_kinds,
                         'plate_barcode': plate_barcodes})
    data.to_csv(sys.argv[2], index=False)
