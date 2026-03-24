import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import streamlit as st
from deepface import DeepFace
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Chamada Inteligente", page_icon="🎓", layout="wide")

PASTA_DB = "banco_de_dados"
if not os.path.exists(PASTA_DB):
    os.makedirs(PASTA_DB)

# --- INICIALIZAÇÃO DO ESTADO (SESSION STATE) ---
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Função para limpar TUDO (nome e foto)
def limpar_tudo():
    st.session_state["nome_aluno"] = ""
    st.session_state["uploader_key"] += 1  # Muda a ID da foto para ela sumir

# --- MENU LATERAL ---
st.sidebar.title("📒 Lista de Presença")
aba = st.sidebar.radio("Navegação:", ["📸 Realizar Chamada", "👤 Cadastrar Alunos"])

# --- ABA: CADASTRAR ALUNO ---
if aba == "👤 Cadastrar Alunos":
    st.header("👤 Cadastro de Novos Alunos")
    
    # Campo de Nome
    nome = st.text_input("Nome Completo:", key="nome_aluno")
    
    # Campo de Foto com KEY DINÂMICA (Isso resolve o seu problema!)
    foto = st.file_uploader("Foto de Identificação", 
                            type=['jpg', 'png', 'jpeg'], 
                            key=f"foto_perfil_{st.session_state['uploader_key']}")
    
    if st.button("Salvar no Banco de Dados"):
        if nome and foto:
            nome_arquivo = nome.lower().strip().replace(' ', '_')
            caminho = os.path.join(PASTA_DB, f"{nome_arquivo}.jpg")
            with open(caminho, "wb") as f:
                f.write(foto.getbuffer())
            
            st.success(f"✅ Aluno '{nome}' cadastrado com sucesso!")
            
            # Limpa cache da IA
            for f in os.listdir(PASTA_DB):
                if f.endswith(".pkl"): os.remove(os.path.join(PASTA_DB, f))
            
            # O BOTÃO QUE LIMPA TUDO NA HORA
            st.button("Cadastrar outro aluno", on_click=limpar_tudo)
        else:
            st.error("⚠️ Por favor, preencha o nome e selecione uma foto.")

# --- ABA: REALIZAR CHAMADA ---
elif aba == "📸 Realizar Chamada":
    st.header("📸 Chamada Inteligente")
    
    arquivo_sala = st.file_uploader("Carregar foto da sala", type=['jpg', 'png', 'jpeg'])
    
    if arquivo_sala:
        img = Image.open(arquivo_sala)
        st.image(img, caption="Visualização da Sala", use_container_width=True)
        
        if st.button("🚀 Iniciar Reconhecimento"):
            with st.spinner("Processando..."):
                img_array = np.array(img)
                cv2.imwrite("temp_sala.jpg", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                
                # Detecção e reconhecimento
                rostos = DeepFace.extract_faces(img_path="temp_sala.jpg", detector_backend='retinaface', enforce_detection=False)
                total_na_foto = len(rostos)
                
                resultados = DeepFace.find(img_path="temp_sala.jpg", db_path=PASTA_DB, detector_backend='retinaface', enforce_detection=False)
                
                nomes_encontrados = set()
                for res in resultados:
                    if not res.empty:
                        nome_cru = os.path.basename(res['identity'][0]).split('.')[0]
                        nomes_encontrados.add(nome_cru.replace('_', ' ').title())
                
                qtd_identificados = len(nomes_encontrados)
                qtd_desconhecidos = max(0, total_na_foto - qtd_identificados)
                agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total de Rostos", total_na_foto)
                c2.metric("Alunos Reconhecidos", qtd_identificados)
                c3.metric("Não Reconhecidos", qtd_desconhecidos)

                if qtd_desconhecidos > 0:
                    st.warning(f"⚠️ Atenção: {qtd_desconhecidos} pessoa(s) na foto não possuem cadastro!")

                if nomes_encontrados:
                    st.subheader(f"📋 Lista de Presença - {agora}")
                    
                    df_chamada = pd.DataFrame({
                        "Nome do Aluno": sorted(list(nomes_encontrados)),
                        "Data": [agora.split()[0]] * qtd_identificados,
                        "Hora": [agora.split()[1]] * qtd_identificados,
                        "Status": ["Presente"] * qtd_identificados
                    })
                    
                    df_chamada.index = np.arange(1, len(df_chamada) + 1)
                    st.table(df_chamada)
                    
                    csv = df_chamada.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="📥 Baixar Lista de Presença (CSV)",
                        data=csv,
                        file_name=f"chamada_{datetime.now().strftime('%d_%m_%Y')}.csv",
                        mime="text/csv",
                    )
                st.markdown("---")
