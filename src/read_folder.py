import os

def get_ful_path(pasta:str='arquivos')->list:
    """
    obtem os arquivos de dentro de um diretorio
    args:
        pasta: (default=arquivos) nome da pasta
    
    return
        lista contendo um dicionário com o caminho e o nome
    """
    lista = []
    for nome in os.listdir(pasta):
        lista.append({"caminho": os.path.join(pasta, nome), "nome":nome.lower()})
    return lista