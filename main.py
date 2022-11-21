import numpy as np


"""
Treliça 2D Linear
"""
# declaração de variáveis
# l = comprimento do elemento
# F = forças aplicadas nos nós indicados

E = 29500000
A = 1
alpha = 0.0000066667
l_e1 = 40
l_e2 = 30
l_e3 = 50
l_e4 = 40
F_q6 = -25000
F_q3 = 20000
delta_t = 50


"""
Conectividade Nodal
Elemento  Nó1  Nó2
   1       1    2
   2       2    3
   3       1    3
   4       3    4


Coordenada Nodal
elemento  le   c    s
   1      40   1    0
   2      30   0    1
   3      50   0.8  0.6
   4      40   -1   0 

"""

"""
(a) Determine o vetor de carregamentos térmicos \n
    de cada elemento no sistema global
(b) Monte o vetor de carregamentos global da estrutura
(c) Use o método direto (eliminação) para resolver \\
    o problema e encontrar os deslocamentos
(d) Determine as tensões em cada elemento (barra)
(e) Calcule os esforços de reação

"""

# Formando a matriz de rigidez global para cada elemento 
# 4 matriz 4x4


def matriz_local_para_global(x_i,x_f,y_i,y_f,Le):

    """
    Realiza a montagem da matriz de rotação da barra de cada elemento \\
    passando do sistema local para o sistema global

    Dados de entrada:
    - x_i = Coord. x do nó inicial da barra [in]
    - y_i = Coord. y do nó inicial da barra [in]
    - x_f = Coord. x do nó inicial da barra [in]
    - y_f = Coord. y do nó inicial da barra [in]
    - Le = Comprimento da barra de cada elemento [in]

    Saída:
    - R = matriz de rotação 4x4 do sistema local para o sistema global
    """

    R = np.zeros((4,4)) #Cria uma matriz 4x4 com 0 em todas as posições

    c = (x_f-x_i)/Le
    s = (y_f-y_i)/Le

    R[0][0] = c**2
    R[0][1] = c*s
    R[0][2] = -c**2
    R[0][3] = -c*s
    R[1][0] = c*s
    R[1][1] = s**2
    R[1][2] = -c*s
    R[1][3] = -s**2
    R[2][0] = -c**2
    R[2][1] = -c*s
    R[2][2] = c**2
    R[2][3] = c*s
    R[3][0] = -c*s
    R[3][1] = -s**2
    R[3][2] = c*s
    R[3][3] = s**2

    return R

#passagem dos parâmetros de comprimento de cada elemento

m_k_elemento1 = matriz_local_para_global(0,40,0,0,40)
m_k_elemento2 = matriz_local_para_global(0,0,0,30,30)
m_k_elemento3 = matriz_local_para_global(0,40,0,30,50)
m_k_elemento4 = matriz_local_para_global(40,0,0,0,40)

"""
print(np.matrix(m_k_elemento1))
print(np.matrix(m_k_elemento2))
print(np.matrix(m_k_elemento3))
print(np.matrix(m_k_elemento4))

"""

# realizando a multiplicação das características do material

caracteristicas_e1 = E/l_e1
caracteristicas_e2 = E/l_e2
caracteristicas_e3 = E/l_e3
caracteristicas_e4 = E/l_e4

m_global_elemento_1 = np.multiply(m_k_elemento1, caracteristicas_e1)
m_global_elemento_2 = np.multiply(m_k_elemento2, caracteristicas_e2)
m_global_elemento_3 = np.multiply(m_k_elemento3, caracteristicas_e3)
m_global_elemento_4 = np.multiply(m_k_elemento4, caracteristicas_e4)


"""
print(m_global_elemento_1)
print(m_global_elemento_2)
print(m_global_elemento_3)
print(m_global_elemento_4)

"""
# Formando a matriz de rigidez global 8x8

def matriz_rigidez_global(m_global_e,n1,n2):

    """
    Realiza o endereçamento da matriz de rigidez da barra i para a matriz de rigidez da estrutura

    Dados de entrada:
    - m_global_e = matriz de rigidez 4x4 da barra de cada elemento no sistema global 
    - n1 = Nó 1
    - n2 = Nó 2
    - k_global = matriz de rigidez 

    Saída:
    - k_global = matriz de rigidez

    """
    k_global = np.zeros((8,8)) #Cria uma matriz 8x8 com 0 em todas as posições

    k_global[2*n1-2][2*n1-2] += m_global_e[0][0]
    k_global[2*n1-2][2*n1-1] += m_global_e[0][1]
    k_global[2*n1-2][2*n2-2] += m_global_e[0][2]
    k_global[2*n1-2][2*n2-1] += m_global_e[0][3]

    k_global[2*n1-1][2*n1-2] += m_global_e[1][0]
    k_global[2*n1-1][2*n1-1] += m_global_e[1][1]
    k_global[2*n1-1][2*n2-2] += m_global_e[1][2]
    k_global[2*n1-1][2*n2-1] += m_global_e[1][3]
    
    k_global[2*n2-2][2*n1-2] += m_global_e[2][0]
    k_global[2*n2-2][2*n1-1] += m_global_e[2][1]
    k_global[2*n2-2][2*n2-2] += m_global_e[2][2]
    k_global[2*n2-2][2*n2-1] += m_global_e[2][3]

    k_global[2*n2-1][2*n1-2] += m_global_e[3][0]
    k_global[2*n2-1][2*n1-1] += m_global_e[3][1]
    k_global[2*n2-1][2*n2-2] += m_global_e[3][2]
    k_global[2*n2-1][2*n2-1] += m_global_e[3][3]  

    return k_global

k_g_e1 = matriz_rigidez_global(m_global_elemento_1,1,2)
k_g_e2 = matriz_rigidez_global(m_global_elemento_2,2,3)
k_g_e3 = matriz_rigidez_global(m_global_elemento_3,1,3)
k_g_e4 = matriz_rigidez_global(m_global_elemento_4,3,4)

#print(np.matrix(k_g_e1))
#print(np.matrix(k_g_e2))
#print(np.matrix(k_g_e3))
#print(np.matrix(k_g_e4))

"""
for linha in k_g_e4:
    for coluna in linha:
        print(coluna,end='  ')
    print('\n')
"""

k_global_total = k_g_e1 + k_g_e2 + k_g_e3 + k_g_e4

#print(np.matrix(k_global_total))

"""
for linha in k_global_total:
    for coluna in linha:
        print(coluna,end='  ')
    print('\n')
"""

indice = 1/49167

k_global_inversa = np.multiply(k_global_total, indice)

#print(np.matrix(k_global_inversa))

"""
for linha in k_global_inversa:
    for coluna in linha:
        print(coluna,end='  ')
    print('\n')
"""

# -----------------------------------------------------------------------------------

 # efeitos térmicos

F_term = E*A*alpha*delta_t

def matriz_local_para_global_termica_tensao(x_i,x_f,y_i,y_f,Le):

    """
    Realiza a montagem da matriz de rotação da barra de cada elemento \\
    passando do sistema local para o sistema global

    Dados de entrada:
    - x_i = Coord. x do nó inicial da barra [in]
    - y_i = Coord. y do nó inicial da barra [in]
    - x_f = Coord. x do nó inicial da barra [in]
    - y_f = Coord. y do nó inicial da barra [in]
    - Le = Comprimento da barra de cada elemento [in]

    Saída:
    - R = matriz de rotação 4x4 do sistema local para o sistema global
    """

    R = np.zeros((1,4)) #Cria uma matriz 1x4 com 0 em todas as posições

    c = (x_f-x_i)/Le
    s = (y_f-y_i)/Le

    R[0][0] = -c
    R[0][1] = -s
    R[0][2] = c
    R[0][3] = s

    return R

m_kt_elemento1 = matriz_local_para_global_termica_tensao(0,40,0,0,40)
m_kt_elemento2 = matriz_local_para_global_termica_tensao(0,0,0,30,30)
m_kt_elemento3 = matriz_local_para_global_termica_tensao(0,40,0,30,50)
m_kt_elemento4 = matriz_local_para_global_termica_tensao(40,0,0,0,40)

# multiplicando por F_term
m_term_elemento_1 = np.multiply(m_kt_elemento1, F_term)
#[[-9833.3825    -0.      9833.3825     0.    ]]
m_term_elemento_2 = np.multiply(m_kt_elemento2, F_term)
#[[   -0.     -9833.3825     0.      9833.3825]]
m_term_elemento_3 = np.multiply(m_kt_elemento3, F_term)
#[[-7866.706  -5900.0295  7866.706   5900.0295]]
m_term_elemento_4 = np.multiply(m_kt_elemento4, F_term)
#[[ 9833.3825    -0.     -9833.3825     0.    ]]

#print("{}".format(m_term_elemento_4))

# --------------------------------------------------------------------------------------------

# formando o vetor de forças

carga_pontual = [0, 0, F_q3, 0, 0, F_q6, 0, 0]

forças_q1 = carga_pontual[0] + m_term_elemento_1[0,0] + m_term_elemento_3[0,0]
forças_q2 = carga_pontual[1] + m_term_elemento_1[0,1] + m_term_elemento_3[0,1]
forças_q3 = carga_pontual[2] + m_term_elemento_1[0,2] + m_term_elemento_2[0,0]
forças_q4 = carga_pontual[3] + m_term_elemento_1[0,3] + m_term_elemento_2[0,1]
forças_q5 = carga_pontual[4] + m_term_elemento_2[0,2] + m_term_elemento_3[0,2] + m_term_elemento_4[0,0]
forças_q6 = carga_pontual[5] + m_term_elemento_2[0,3] + m_term_elemento_3[0,3] + m_term_elemento_4[0,1]
forças_q7 = carga_pontual[6] + m_term_elemento_4[0,2]
forças_q8 = carga_pontual[7] + m_term_elemento_4[0,3]

#print("{}".format(forças_q6))

vetor_carregamentos_termicos_pontuais = [forças_q1, forças_q2, forças_q3, forças_q4, 
forças_q5, forças_q6, forças_q7, forças_q8]

#print("{}".format(vetor_carregamentos_termicos_pontuais))

# ----> considerando o vetor com o método direto
# q1 = 0, q2 = 0, q4 = 0, q7 = 0, q8 = 0 (todos os nós que possuem engastes)

vetor_carregamentos_termicos_pontuais_MD = [0, 0, forças_q3, 0, 
forças_q5, forças_q6, 0, 0]

#print("{}".format(vetor_carregamentos_termicos_pontuais_MD))

# ------------------------------------------------------------------------------

#aplicando carregamentos térmicos apenas nos elementos 2 e 3
#seguindo instruções do livro

# formando o vetor de forças

forças_q1_2 = m_term_elemento_1[0,0] + m_term_elemento_3[0,0]
forças_q2_2 = m_term_elemento_1[0,1] + m_term_elemento_3[0,1]
forças_q3_2 = m_term_elemento_2[0,0] 
forças_q4_2 = m_term_elemento_1[0,3] + m_term_elemento_2[0,1]
forças_q5_2 = m_term_elemento_2[0,2] + m_term_elemento_3[0,2]
forças_q6_2 = m_term_elemento_2[0,3] + m_term_elemento_3[0,3]
forças_q7_2 = m_term_elemento_4[0,2]
forças_q8_2 = m_term_elemento_4[0,3]

#print("{}".format(forças_q6))

vetor_carregamentos_termicos_pontuais_2 = [forças_q1_2, forças_q2_2, forças_q3_2, forças_q4_2, 
forças_q5_2, forças_q6_2, forças_q7_2, forças_q8_2]

#print("{}".format(vetor_carregamentos_termicos_pontuais))

# ----> considerando o vetor com o método direto
# q1 = 0, q2 = 0, q4 = 0, q7 = 0, q8 = 0 (todos os nós que possuem engastes)

vetor_carregamentos_termicos_pontuais_MD_2 = [0, 0, forças_q3_2, 0, 
forças_q5_2, forças_q6_2, 0, 0]

#print("{}".format(vetor_carregamentos_termicos_pontuais_MD_2))


#SAÍDA

"""
[0.         0.         0.         0.         0.00395064 0.01222228
 0.         0.        ]

q3 = 0
q5 = 0.00395064
q6 = 0.01222228

"""

# ------------------------------------------------------------------------------

"""
---> Aplicando o método direto

A treliça apenas possui carregamentos nos nós q3 e q6

"""
#elemento 1
# q1 - coluna 0 linha 0
# q2 - coluna 1 linha 1
# zerando coluna 1 e linha 1 
# zerando coluna 2 e linha 2 

# el = [linha][coluna]

k_global_total[:,0] = 0 # zera a coluna 0
k_global_total[:,1] = 0 # zera a coluna 1
k_global_total[0,:] = 0 # zera a linha 0
k_global_total[1,:] = 0 # zera a linha 1

# inserindo 1 na diagonal principal
k_global_total[0,0] = 1
k_global_total[1,1] = 1

#elemento 2
# q3 - possui carregamentos
# q4 - coluna 3 linha 3
# zerando coluna 3 e linha 3

k_global_total[:,3] = 0 # zera a coluna 0
k_global_total[3,:] = 0 # zera a linha 0

# inserindo 1 na diagonal principal
k_global_total[3,3] = 1

#elemento 3
# q5 - não possui suporte fixo
# q6 - possui carregamentos

#elemento 4
# q7 - coluna 6 linha 6
# q8 - coluna 7 linha 7
# zerando coluna 6 e linha 6 
# zerando coluna 7 e linha 7 

# el = [linha][coluna]

k_global_total[:,6] = 0 # zera a coluna 6
k_global_total[:,7] = 0 # zera a coluna 7
k_global_total[6,:] = 0 # zera a linha 6
k_global_total[7,:] = 0 # zera a linha 7

# inserindo 1 na diagonal principal
k_global_total[6,6] = 1
k_global_total[7,7] = 1


# imprimindo a matriz

"""
for linha in k_global_total:
    for coluna in linha:
        print(coluna,end='  ')
    print('\n')

"""

# ---------------------------------------------------------------------

# Obtendo os deslocamentos

inversa_k_global = np.linalg.inv(k_global_total)

Q = np.dot(inversa_k_global,vetor_carregamentos_termicos_pontuais_MD)

#print("{}".format(Q))

# Obtendo os deslocamentos para carregamentos térmicos 
# apenas nos elementos 2 e 3, sem cargas pontuais

Q_sem_cargas_pontuais = np.dot(inversa_k_global,vetor_carregamentos_termicos_pontuais_MD_2)

#print("{}".format(Q_sem_cargas_pontuais))

# --------------------------------------------------------------------------------

#calculando tensoes - utilizando cargas pontuais e cargas térmicas

deslocamento_n1 = Q[0]
deslocamento_n2 = Q[1]
deslocamento_n3 = Q[2]
deslocamento_n4 = Q[3]
deslocamento_n5 = Q[4]
deslocamento_n6 = Q[5]
deslocamento_n7 = Q[6]
deslocamento_n8 = Q[7]

vetor_q1 = [deslocamento_n1, deslocamento_n2, deslocamento_n3, deslocamento_n4]
vetor_q2 = [deslocamento_n3, deslocamento_n4, deslocamento_n5, deslocamento_n6]
vetor_q3 = [deslocamento_n1, deslocamento_n2, deslocamento_n5, deslocamento_n6]
vetor_q4 = [deslocamento_n5, deslocamento_n6, deslocamento_n7, deslocamento_n8]

indice_sigma_e1 = E/l_e1
indice_sigma_e2 = E/l_e2
indice_sigma_e3 = E/l_e3
indice_sigma_e4 = E/l_e4

sigma_e1_indice = np.multiply(m_kt_elemento1, indice_sigma_e1)
sigma_e1_deslocamento = np.multiply(sigma_e1_indice,vetor_q1)
sigma_e1 = sigma_e1_deslocamento[0,2] - F_term

# SAÍDA 20 000

sigma_e2_indice = np.multiply(m_kt_elemento2, indice_sigma_e2)
sigma_e2_deslocamento = np.multiply(sigma_e2_indice,vetor_q2)
sigma_e2_q3 = sigma_e2_deslocamento[0,0] 
sigma_e2_q6 = sigma_e2_deslocamento[0,3] 
sigma_e2_q5 = sigma_e2_deslocamento[0,2]
sigma_e2_soma = sigma_e2_q3 + sigma_e2_q6 + sigma_e2_q5
sigma_e2 = sigma_e2_soma - F_term

#SAÍDA -21874.999999999996

sigma_e3_indice = np.multiply(m_kt_elemento3, indice_sigma_e3)
sigma_e3_deslocamento = np.multiply(sigma_e3_indice,vetor_q3)
sigma_e3_q6 = sigma_e3_deslocamento[0,3] 
sigma_e3_q5 = sigma_e3_deslocamento[0,2]
sigma_e3_soma = sigma_e3_q6 + sigma_e3_q5
sigma_e3 = sigma_e3_soma - F_term

#SAÍDA -5208.333333333331

sigma_e4_indice = np.multiply(m_kt_elemento4, indice_sigma_e4)
sigma_e4_deslocamento = np.multiply(sigma_e4_indice,vetor_q4)
sigma_e4_q6 = sigma_e4_deslocamento[0,1] 
sigma_e4_q5 = sigma_e4_deslocamento[0,0]
sigma_e4_soma = sigma_e4_q6 + sigma_e4_q5
sigma_e4 = sigma_e4_soma - F_term

#SAÍDA 4166.666666666668

#print("{}".format(sigma_e4))

#-------------------------------------------------
#calculando tensoes - sem cargas pontuais 
# e com cargas térmicas apenas nos elementos 2 e 3

deslocamento_n1_2 = Q_sem_cargas_pontuais[0]
deslocamento_n2_2 = Q_sem_cargas_pontuais[1]
deslocamento_n3_2 = Q_sem_cargas_pontuais[2]
deslocamento_n4_2 = Q_sem_cargas_pontuais[3]
deslocamento_n5_2 = Q_sem_cargas_pontuais[4]
deslocamento_n6_2 = Q_sem_cargas_pontuais[5]
deslocamento_n7_2 = Q_sem_cargas_pontuais[6]
deslocamento_n8_2 = Q_sem_cargas_pontuais[7]

vetor_q1_2 = [deslocamento_n1_2, deslocamento_n2_2, deslocamento_n3_2, deslocamento_n4_2]
vetor_q2_2 = [deslocamento_n3_2, deslocamento_n4_2, deslocamento_n5_2, deslocamento_n6_2]
vetor_q3_2 = [deslocamento_n1_2, deslocamento_n2_2, deslocamento_n5_2, deslocamento_n6_2]
vetor_q4_2 = [deslocamento_n5_2, deslocamento_n6_2, deslocamento_n7_2, deslocamento_n8_2]


sigma_e1_indice_2 = np.multiply(m_kt_elemento1, indice_sigma_e1)
sigma_e1_deslocamento_2 = np.multiply(sigma_e1_indice_2,vetor_q1_2)
sigma_e1_2 = sigma_e1_deslocamento_2[0,2]

# SAÍDA 0

sigma_e2_indice_2 = np.multiply(m_kt_elemento2, indice_sigma_e2)
sigma_e2_deslocamento_2 = np.multiply(sigma_e2_indice_2,vetor_q2_2)
sigma_e2_q3_2 = sigma_e2_deslocamento_2[0,0] 
sigma_e2_q6_2 = sigma_e2_deslocamento_2[0,3] 
sigma_e2_q5_2 = sigma_e2_deslocamento_2[0,2]
sigma_e2_soma_2 = sigma_e2_q3_2 + sigma_e2_q6_2 + sigma_e2_q5_2
sigma_e2_2 = sigma_e2_soma_2 - F_term

#SAÍDA 2185.1961111111104

sigma_e3_indice_2 = np.multiply(m_kt_elemento3, indice_sigma_e3)
sigma_e3_deslocamento_2 = np.multiply(sigma_e3_indice_2,vetor_q3_2)
sigma_e3_q6_2 = sigma_e3_deslocamento_2[0,3] 
sigma_e3_q5_2 = sigma_e3_deslocamento_2[0,2]
sigma_e3_soma_2 = sigma_e3_q6_2 + sigma_e3_q5_2
sigma_e3_2 = sigma_e3_soma_2 - F_term

#SAÍDA -3641.9935185185177

sigma_e4_indice_2 = np.multiply(m_kt_elemento4, indice_sigma_e4)
sigma_e4_deslocamento_2 = np.multiply(sigma_e4_indice_2,vetor_q4_2)
sigma_e4_q6_2 = sigma_e4_deslocamento_2[0,1] 
sigma_e4_q5_2 = sigma_e4_deslocamento_2[0,0]
sigma_e4_soma_2 = sigma_e4_q6_2 + sigma_e4_q5_2
sigma_e4_2 = sigma_e4_soma_2 

#SAÍDA 2913.594814814816

#print("{}".format(sigma_e3_2))


# ----------------------------------------------------------------

#Calculando forças de reação
# R = K*Q - F

# considerando cargas pontuais e térmicas

#utilizando a matriz de rigidez global sem o método direto
k_global_total_semMD = k_g_e1 + k_g_e2 + k_g_e3 + k_g_e4

r = np.multiply(k_global_total_semMD,Q)
r1_soma = r[0,1] + r[0,2] + r[0,3] + r[0,4] + r[0,5] + r[0,6] + r[0,7]
r1 = r1_soma - vetor_carregamentos_termicos_pontuais[0]
 
#SAÍDA -15833.333333333336

r2_soma = r[1,1] + r[1,2] + r[1,3] + r[1,4] + r[1,5] + r[1,6] + r[1,7]
r2 = r2_soma - vetor_carregamentos_termicos_pontuais[1]

#SAÍDA 3124.999999999998

r4_soma = r[3,1] + r[3,2] + r[3,3] + r[3,4] + r[3,5] + r[3,6] + r[3,7]
r4 = r4_soma - vetor_carregamentos_termicos_pontuais[3]

#SAÍDA 21874.999999999996

r7_soma = r[6,1] + r[6,2] + r[6,3] + r[6,4] + r[6,5] + r[6,6] + r[6,7]
r7 = r7_soma - vetor_carregamentos_termicos_pontuais[6]

#SAÍDA -4166.666666666668

r8_soma = r[7,1] + r[7,2] + r[7,3] + r[7,4] + r[7,5] + r[7,6] + r[7,7]
r8 = r8_soma - vetor_carregamentos_termicos_pontuais[7]

#SAÍDA 0.0

#print("{}".format(r8))

# -------------------------
#desconsiderando cargas pontuais e considerando
# térmicas apenas nos elementos 2 e 3

r_2 = np.multiply(k_global_total_semMD,Q_sem_cargas_pontuais)
r1_soma_2 = r_2[0,1] + r_2[0,2] + r_2[0,3] + r_2[0,4] + r_2[0,5] + r_2[0,6] + r_2[0,7]
r1_2 = r1_soma_2
#SAÍDA -4953.111185185186

r2_soma_2 = r_2[1,1] + r_2[1,2] + r_2[1,3] + r_2[1,4] + r_2[1,5] + r_2[1,6] + r_2[1,7]
r2_2 = r2_soma_2

#SAÍDA -3714.8333888888887

r4_soma_2 = r_2[3,1] + r_2[3,2] + r_2[3,3] + r_2[3,4] + r_2[3,5] + r_2[3,6] + r_2[3,7]
r4_2 = r4_soma_2

#SAÍDA -12018.57861111111

r7_soma_2 = r_2[6,1] + r[6,2] + r_2[6,3] + r_2[6,4] + r_2[6,5] + r_2[6,6] + r_2[6,7]
r7_2 = r7_soma_2

#SAÍDA -2913.594814814816

r8_soma_2 = r_2[7,1] + r_2[7,2] + r_2[7,3] + r_2[7,4] + r_2[7,5] + r_2[7,6] + r_2[7,7]
r8_2 = r8_soma_2

#SAÍDA 0.0

#print("{}".format(r8_2))


with open('resultados.md', 'w') as arquivo:
    arquivo.write('Modulo de Young: {}\n Area: {}\n Coeficiente de dilatacao: {}\n Comprimento barra 1: {}\n Comprimento barra 2: {}\n Comprimento barra 3: {}\n Comprimento barra 4: {}\n Carregamento no 6: {}\n Carregamento no 3: {}\n Variacao de Temperatura: {}\n  Matriz global e1: {}\n Matriz global e2: {}\n Matriz global e3: {}\n Matriz global e4: {}\n Matriz de rigidez global: {}\n Matriz de rigidez total apos metodo direto: {}\n Vetor de carregamentos termicos com cargas pontuais: {}\n Vetor de carregamentos termicos sem cargas pontuais: {}\n Deslocamentos: {}\n Deslocamentos sem cargas pontuais: {}\n Sigma e1: {}\n Sigma e2: {}\n Sigma e3: {}\n Sigma e4:{}\n Sigma e1 sem cargas pontuais: {}\n Sigma e2 sem cargas pontuais: {}\n Sigma e3 sem cargas pontuais: {}\n Sigma e4 sem cargas pontuais: {}\n Reacao 1: {}\n Reacao 2: {}\n Reacao 4: {}\n Reacao 7: {}\n Reacao 8: {}\n Reacao 1 sem carga pontual: {}\n Reacao 2 sem carga pontual: {}\n Reacao 4 sem carga pontual: {}\n Reacao 7 sem carga pontual: {}\n Reacao 8 sem carga pontual: {}\n'.format(
        E,
        A,
        alpha,
        l_e1,
        l_e2,
        l_e3,
        l_e4,
        F_q6,
        F_q3,
        delta_t,
        m_global_elemento_1,
        m_global_elemento_2,
        m_global_elemento_3,
        m_global_elemento_4,
        k_global_total_semMD,
        k_global_total,
        vetor_carregamentos_termicos_pontuais_MD,
        vetor_carregamentos_termicos_pontuais_MD_2,
        Q,
        Q_sem_cargas_pontuais,
        sigma_e1,
        sigma_e2,
        sigma_e3,
        sigma_e4,
        sigma_e1_2,
        sigma_e2_2,
        sigma_e3_2,
        sigma_e4_2,
        r1,
        r2,
        r4,
        r7,
        r8,
        r1_2,
        r2_2,
        r4_2,
        r7_2,
        r8_2
    ))

