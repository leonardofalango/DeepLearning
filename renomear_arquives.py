import os

def renomear_arquivos(caminho_pasta, novo_prefixo):
    # Obtém a lista de arquivos na pasta
    lista_arquivos = os.listdir(caminho_pasta)

    # Itera sobre cada arquivo na lista
    for nome_arquivo in lista_arquivos:
        # Constrói o caminho completo para o arquivo
        caminho_completo_antigo = os.path.join(caminho_pasta, nome_arquivo)

        # Verifica se é um arquivo (não é uma pasta)
        if os.path.isfile(caminho_completo_antigo):
            # Separa o nome do arquivo e a extensão
            nome_base, extensao = os.path.splitext(nome_arquivo)

            # Constrói o novo nome do arquivo
            novo_nome = f"{novo_prefixo}_{nome_base}{extensao}"

            # Constrói o caminho completo para o novo arquivo
            caminho_completo_novo = os.path.join(caminho_pasta, novo_nome)

            # Renomeia o arquivo
            os.rename(caminho_completo_antigo, caminho_completo_novo)
            print(f"Arquivo renomeado: {nome_arquivo} -> {novo_nome}")

# Exemplo de uso
caminho_pasta = "C:/Users/disrct/Desktop/LeonardoFalango/DeepLearning/Micro_Expressions/test/surprise"  # Substitua pelo caminho real da sua pasta
novo_prefixo = "1"
renomear_arquivos(caminho_pasta, novo_prefixo)
