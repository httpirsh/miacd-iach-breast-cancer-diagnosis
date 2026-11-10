# Fontes dos documentos do RAG

Documentos médicos usados pelo chatbot do agente (notebook `04_agente.ipynb`, `src/rag.py`). Todos são de entidades públicas ou sem fins lucrativos (SNS 24, DGS, Liga Portuguesa Contra o Cancro, hospitais do SNS, Sociedade Portuguesa de Radiologia e Medicina Nuclear), em português de Portugal. Não há fontes comerciais, fóruns nem blogs.

Consultados a 10/11/2026. Este ficheiro não é lido pelo chatbot.

## Como foram preparados

- **Páginas web** (SNS 24, LPCC, DGS): texto copiado da página, sem menus, botões e ligações para outras páginas. O texto não foi alterado. Nas páginas da LPCC sobre diagnóstico e métodos de tratamento foram retiradas as listas de perguntas que a pessoa pode fazer ao médico: são só perguntas, sem respostas, e na pesquisa apareciam à frente dos excertos que respondem (uma pergunta do paciente parece-se mais com outra pergunta do que com a resposta).
- **PDFs** (folhetos dos hospitais, norma da DGS, artigo da EUSOBI): passados a `.md`, porque o texto extraído dos PDFs vinha com as colunas trocadas e palavras cortadas. Os folhetos foram copiados por ordem de leitura; foram retirados contactos telefónicos, instruções práticas próprias de cada hospital e a indicação do analgésico a tomar (o chatbot não recomenda medicamentos). A norma da DGS é um excerto (as regras de registo e os indicadores ficaram de fora). Cada ficheiro diz no início o que foi retirado.
- Cada ficheiro começa com o título, a entidade e o endereço de onde foi tirado, para o chatbot poder indicar a fonte.

## Lista

| Ficheiro | Título | Entidade | Endereço | Público-alvo |
|---|---|---|---|---|
| `sns24_rastreio_cancro_mama.md` | Rastreio do cancro da mama | SNS 24 | https://www.sns24.gov.pt/pt/tema/prevencao-e-cuidados-de-saude/rastreio-do-cancro-da-mama/ | paciente |
| `sns24_mamografia.md` | Mamografia | SNS 24 | https://www.sns24.gov.pt/pt/tema/exames-e-tratamentos/mamografia/ | paciente |
| `sns24_cancro_o_que_e.md` | Cancro (o que é, tumores benignos e malignos) | SNS 24 | https://www.sns24.gov.pt/pt/tema/doencas/doencas-oncologicas/cancro/ | paciente |
| `dgs_perguntas_respostas_cancro.md` | Perguntas e respostas sobre o cancro | DGS, Programa Nacional para as Doenças Oncológicas | https://www.dgs.pt/paginas-de-sistema/saude-de-a-a-z/programa-nacional-para-as-doencas-oncologicas/perguntas-e-respostas.aspx | paciente |
| `dgs_norma_012_2024_rastreio.md` | Norma n.º 012/2024: Programa de rastreio de base populacional do cancro da mama (excerto) | DGS | https://www.dgs.pt/normas-orientacoes-e-informacoes/normas-e-circulares-normativas/norma-122024-de-06122024-programa-de-rastreio-de-base-populacional-do-cancro-da-mama-pdf.aspx | médico |
| `lpcc_cancro_da_mama.md` | O cancro da mama | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama/ | paciente |
| `lpcc_sintomas.md` | Cancro da mama: sintomas de alerta | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-sintomas/ | paciente |
| `lpcc_deteccao.md` | Cancro da mama: deteção | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-deteccao/ | paciente |
| `lpcc_diagnostico.md` | Cancro da mama: diagnóstico | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-diagnostico/ | paciente |
| `lpcc_fatores_de_risco.md` | Cancro da mama: fatores de risco | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-factores-de-risco/ | paciente |
| `lpcc_estadiamento.md` | Cancro da mama: estadiamento | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-estadiamento/ | paciente |
| `lpcc_tratamento.md` | Cancro da mama: o tratamento | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-tratamento/ | paciente |
| `lpcc_metodos_de_tratamento.md` | Cancro da mama: métodos de tratamento | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-metodos-de-tratamento/ | paciente |
| `lpcc_efeitos_secundarios.md` | Cancro da mama: efeitos secundários do tratamento | Liga Portuguesa Contra o Cancro | https://www.ligacontracancro.pt/cancro-da-mama-efeitos-secundarios/ | paciente |
| `chsj_mamografia_ecografia.md` | Mamografia e ecografia mamária: recomendações ao doente (2020) | Centro de Mama, Centro Hospitalar Universitário de São João (SNS) | https://portal-chsj.min-saude.pt/portalchsj/uploads/writer_file/document/5269/CMA-IM025-1_Informacao_doente_Mamografia_ecografia_mamaria.pdf | paciente |
| `chsj_biopsia_mamaria_por_agulha.md` | Biópsia mamária por agulha: microbiópsia e citologia aspirativa (2020; junta dois folhetos) | Centro de Mama, Centro Hospitalar Universitário de São João (SNS) | https://portal-chsj.min-saude.pt/portalchsj/uploads/writer_file/document/5277/CMA-IM024-1_Informacao_doente_Biopsia_Mamaria_por_agulha.pdf e https://portal-chsj.min-saude.pt/portalchsj/uploads/writer_file/document/5265/CMA-IM043-1_Informacao_doente_Biopsia_Mamaria_Citologia_Aspirativa.pdf | paciente |
| `chsj_biopsia_excisional.md` | Biópsia excisional de nódulo da mama (2020) | Centro de Mama, Centro Hospitalar Universitário de São João (SNS) | https://portal-chsj.min-saude.pt/portalchsj/uploads/writer_file/document/5275/CMA-IM068-0_Informcacao_doente_Biopsia_Excisional.pdf | paciente |
| `ulssm_fibroadenoma.md` | Tratamento do fibroadenoma mamário | Cirurgia de Ambulatório, CHU Lisboa Norte (atual ULS Santa Maria, SNS) | https://www.ulssm.min-saude.pt/media/k2/attachments/unidade_cirurgia_ambulatorio/I7396%20Folh%20CHLN%20-%20Nodulo%20da%20Mama_7.pdf | paciente |
| `sprmn_eusobi_mamografia.md` | Mamografia: atualização das recomendações da EUSOBI para informação à mulher (2017) | Sociedade Portuguesa de Radiologia e Medicina Nuclear (tradução do artigo da European Society of Breast Imaging) | https://www.sprmn.pt/pdf/EUSOBI_Mammography_1_Traduc_a_o.pdf | paciente e médico |

## Temas cobertos

- **Rastreio e mamografia:** SNS 24, DGS (norma), LPCC (deteção), EUSOBI.
- **Ecografia e o que significa cada resultado (BI-RADS):** São João (mamografia e ecografia), DGS (norma), EUSOBI.
- **Benigno e maligno:** SNS 24 (cancro), LPCC (diagnóstico), ULS Santa Maria (fibroadenoma), São João (biópsia excisional).
- **Biópsia e próximos passos:** São João (biópsias), LPCC (diagnóstico), DGS (norma).
- **Limites dos exames (falsos negativos, falsos positivos, sobrediagnóstico):** EUSOBI, São João (mamografia e ecografia).
- **Sintomas e fatores de risco:** LPCC.
- **Se for cancro (estadio, tratamentos, efeitos secundários):** LPCC.

## Limitações

- Não encontrámos um texto público para doentes, em português de Portugal, sobre a ecografia mamária sozinha nem sobre as categorias BI-RADS da ecografia. O BI-RADS é explicado no artigo da EUSOBI (sobre mamografia) e na norma da DGS (sobre o rastreio).
- Algumas páginas da LPCC usam a ortografia anterior ao Acordo Ortográfico (ex.: "detecção", "factores"), e os folhetos do São João são de 2020.
- O folheto do São João refere rastreio dos 40 aos 75 anos; o programa nacional atual é dos 45 aos 74 (Norma 012/2024 da DGS). O ficheiro tem uma nota a dizer isto.
