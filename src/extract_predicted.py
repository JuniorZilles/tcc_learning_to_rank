import pandas as pd
from read_folder import get_ful_path

def extract_predicted():
    paths = get_ful_path('predicted_csv')
    for path in paths:
        f_name = path['nome']
        spl_name = f_name.split('.')
        model = spl_name[0]
        algoritmo = spl_name[1]
        dataset = spl_name[2]
        df = pd.read_csv(path['caminho'])
        ndf =  df['predicted_ranking'].astype(str)
        pred =ndf.tolist()
        with open(f'predicted/{model}.{algoritmo}.{dataset}.txt', 'w') as outfile:
            outfile.write("\n".join(pred))

extract_predicted()