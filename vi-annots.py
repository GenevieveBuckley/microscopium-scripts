# IPython log file
import numpy as np
import pandas as pd
import itertools
from glob import glob
annots = pd.concat([pd.read_csv(fn) for fn in glob('kirsty-compound-annotation/*Batch*.csv')], axis=0)
plate_codes = pd.read_csv('PlateID_UniqueIdentifier.csv')
plate_mid = pd.read_csv('kirsty_plates.csv')
late_annots = pd.concat([pd.read_csv(fn, sep='\t') for fn in glob('kirsty-compound-annotation/*.tsv')], axis=0)
controls = pd.read_csv('kirsty-compound-annotation/Annotation_Kirsty_Controls.csv')

annots['concentration'] = annots['concentration'].astype(str)
annots['summary'] = annots['libary.id'].str.cat(annots[['sample.no', 'concentration']], sep=' - ')
annots.rename(columns={'libary.id': 'library', 'sample.no': 'compound'}, inplace=True)
annots['coord'] = annots['plate.id'].str.cat(annots['well'], sep='-')

late_annots.dropna(axis=0, subset=['PURPOSE'], inplace=True)
late_annots['concentration'] = late_annots['CONC_SUPPLIED'].astype(int).astype(str)
late_annots['summary'] = late_annots['PURPOSE'].str.cat(late_annots[['QCL_SAMPLE_NUMBER', 'concentration']], sep=' - ')
late_annots.rename(columns={'PURPOSE': 'library', 'QCL_SAMPLE_NUMBER': 'compound', 'PLATE_ID': 'plate.id'}, inplace=True)
late_annots['coord'] = late_annots['plate.id'].str.cat(late_annots['TARGET_COORDINATES'], sep='-')

cols_to_keep = ['coord', 'summary', 'library', 'compound', 'concentration']
annots2 = pd.concat([annots[cols_to_keep],
                     late_annots[cols_to_keep]], axis=0)
annots2.set_index('coord', inplace=True)

barcode2plateid = dict(zip(plate_mid['plate_barcode'], plate_mid['plate_kind']))
plate_codes.replace({'20170924-KTC-07A rescan': '20170924-KTC-07A'}, inplace=True)
desc2barcode = dict(zip(plate_codes['Unique Plate Descriptor'], plate_codes['PlateId/Barcode']))
barcode2kind = dict(zip(plate_mid['plate_barcode'], plate_mid['plate_kind']))
desc2kind = {k: barcode2kind[desc2barcode[k]] for k in desc2barcode}

def coord2coord(coord):
    desc, well = coord.split('-')
    row, col = well[0], well[1:]
    if col in ['23', '24']:
        return None
    else:
        return desc2kind[desc] + '-' + well
    
all_plates = list(desc2kind.keys())
rows = [chr(n + ord('A')) for n in range(16)]
cols = [f'{i:02}' for i in range(1, 25)]
wells = [''.join([row, col]) for row in rows for col in cols]
all_coords = list(itertools.product(all_plates, wells))

def conc(astr):
    if astr.endswith('uM'):
        return astr.split('_')[1]
    else:
        return 'NA'
    
controls['concentration'] = controls['Compound_ID'].apply(conc)
controls['compound'] = controls['Compound_ID'].apply(lambda x: x.split('_')[0])
controls['library'] = controls['Type']
controls['summary'] = controls['library'].str.cat(controls[['compound', 'concentration']], sep=' - ')

controls2 = controls[['Well_ID', 'summary', 'library', 'compound', 'concentration']].rename(columns={'Well_ID': 'well'}).set_index('well')

empty_df = pd.DataFrame({'summary': [None], 'library': [None], 'compound': [None], 'concentration': [None]})
dfs = []

for plate, well in all_coords:
    coord = '-'.join([plate, well])
    dfcoo = coord2coord(coord)
    if dfcoo is not None:
        try:
            dfs.append(annots2.loc[dfcoo].to_frame(name=coord).T)
        except KeyError:
            empty_df.index = [coord]
            dfs.append(empty_df.copy())
    else:
        dfs.append(controls2.loc[well].to_frame(name=coord).T)
        
final_info = pd.concat(dfs)
final_info.to_csv('info.csv', index=False)
