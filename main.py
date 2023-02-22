import time
import math
import random
import sys
import threading
import networkx as nx
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000000)
threading.stack_size(2**26)

########################################################################################################
########## Funkcje tworzace grafy, wypisujace macierze i wpisujace je do pliku ( funkcje pomocnicze) ###

dfs = []  # macierz odwiedzin
stacklist = []
oddnodes = 0


# algorytmy dla maciezy sasiedztwa
def dfs_search_neighbourhood(node, res, node_quan, matrix):
    global dfs
    global stacklist
    global oddnodes
    neighbours = 0
    if dfs[node] is False:
        res.append(node)
        dfs[node] = True
        stacklist[node] = True
        for i in range(node_quan):
            if matrix[node][i] == 1:
                neighbours += 1
            if matrix[node][i] == 1 and dfs[i] is False:
                dfs_search_neighbourhood(i, res, node_quan, matrix)
    if neighbours % 2 == 1:
        oddnodes += 1
    stacklist[node] = False


# tworzenie losowego grafu nieskierowanego
def createRandomGraph(n, p):
    g = nx.gnp_random_graph(n, p)
    data = [(n, len(g.edges))]
    pos = nx.spring_layout(g)

    return data + list(g.edges)


# tworzenie losowego grafu "skierowanego"
def createrandomDiGraph(nodes, source, proportion):
    graph = nx.gnp_random_graph(nodes, proportion)
    pos = nx.spring_layout(graph)

    name = str(source)
    plik = open(name, 'w')
    plik.write(str(nodes) + ' ')
    plik.write(str(len(graph.edges)) + '\n')
    for edge in graph.edges:
        r = random.randint(0, 1)
        if r == 0:
            plik.write(str(edge[0]) + ' ')
            plik.write(str(edge[1]) + '\n')
        else:
            plik.write(str(edge[1]) + ' ')
            plik.write(str(edge[0]) + '\n')
    plik.close()


# zapisywanie grafu nieskierowanego do pliku
def savetofile(file, graph):
    source = open(file, 'w')
    nodes = graph[0][0]
    edges = graph[0][1]

    source.write(str(nodes) + ' ')
    source.write(str(edges))

    for i in range(1,edges + 1):
        source.write('\n' + str(graph[i][0]) + ' ' + str(graph[i][1]))
    source.close()


# tworzenie listy nastepnikow dla grafu skierowanego
def create_nextto(source_path):
    source = open(source_path, "r")
    first_line = source.readline().split(" ")
    nodes_quantity = int(first_line[0])
    edges_quantity = int(first_line[1])
    res = []

    for i in range(nodes_quantity):
        res.append([])

    for edge in range(edges_quantity):
        line = source.readline().split(" ")
        exit_node = int(line[0])
        entry_node = int(line[1])
        res[exit_node].append(entry_node)

    for i in range(nodes_quantity):
        res[i].sort()

    source.close()
    return res


# tworzenie macierzy sasiedztwa dla grafu nieskierowanego
def create_neighbourhood_matrix(source_path):
    source = open(source_path, "r")
    first_line = source.readline().split(' ')
    nodes_quantity = int(first_line[0])
    edges_quantity = int(first_line[1])

    n_matrix = []
    for line in range(nodes_quantity):
        col = []
        for columns in range(nodes_quantity):
            col.append(0)
        n_matrix.append(col)

    for edge in range(edges_quantity):
        line = source.readline().split(' ')
        node1 = int(line[0])
        node2 = int(line[1])
        n_matrix[node1][node2] = 1
        n_matrix[node2][node1] = 1

    source.close()
    return n_matrix


# wypisywanie macierzy
def print_matrix(matrix):
    for i in matrix:
        print(i)
###################################################
###################################################

########## Algorytmy znajdujace cykle hamiltona ###
# zainicjowanie pamieci dla cyklu hamiltona
hamcycle = []

#ilosc odwiedzonych wierzcholkow
visited = 0

# zainicjowanie pamieci dla wierzcholkow odwiedzonych
vis_tab = []

# Algorytm dla grafu nieskierowanego
def rf_search_nontargeted(matrix, node):
    global hamcycle
    global visited
    global vis_tab

    vis_tab[node] = True
    visited += 1

    for i in range(len(matrix)):
        if matrix[node][i] == 1:
            if i == 0 and visited == len(matrix):
                return True
            if not vis_tab[i]:
                if rf_search_nontargeted(matrix, i):
                    hamcycle.append(i)
                    return True

    vis_tab[node] = False
    visited -= 1
    return False


def rf_algorithm_nontargeted(matrix):
    global hamcycle
    global visited
    global vis_tab

    hamcycle = [0]
    visited = 0
    vis_tab = []

    for i in range(len(matrix)):
        vis_tab.append(False)

    cycle = rf_search_nontargeted(matrix, 0)
    if cycle:
        res = []
        hamcycle.append(0)
        for i in range(len(hamcycle) - 1, -1, -1):
            res.append(hamcycle[i])
        return res
    else:
        return "Cykl hamiltona nie istnieje"


# algorytm dla grafu skierowanego
def rf_search_targeted(matrix, node):
    global hamcycle
    global visited
    global vis_tab

    vis_tab[node] = True
    visited += 1

    for i in matrix[node]:
        if i == 0 and visited == len(matrix):
            return True
        if not vis_tab[i]:
            if rf_search_targeted(matrix, i):
                hamcycle.append(i)
                return True

    vis_tab[node] = False
    visited -= 1
    return False


def rf_algorithm_targeted(matrix):
    global hamcycle
    global visited
    global vis_tab

    hamcycle = [0]
    visited = 0
    vis_tab = []

    for i in range(len(matrix)):
        vis_tab.append(False)

    cycle = rf_search_targeted(matrix, 0)
    if cycle:
        res = []
        hamcycle.append(0)
        for i in range(len(hamcycle) - 1, -1, -1):
            res.append(hamcycle[i])
        return res
    else:
        return "Cykl hamiltona nie istnieje"


#############################################
############# Algorytmy znajdujace cykle eulera ##

# dla grafu nieskierowanego
def possible(matrix):
    global dfs
    global stacklist
    global oddnodes
    oddnodes = 0
    dfs = []
    stacklist = []

    nodes_quantity = len(matrix)

    # tworzenie tablicy odwiedzonych wierzcholkow
    for i in range(nodes_quantity):
        dfs.append(False)
    # tworzenie tablicy rekurencji
    for i in range(nodes_quantity):
        stacklist.append(False)

    # odwiedzanie wierzcholkowi sprawdzenie mozliwosci wystapienie cyklu eulera
    res = []
    iteration = 0
    while len(res) != nodes_quantity:
        dfs_search_neighbourhood(iteration, res, nodes_quantity, matrix)
        iteration += 1
        for i in range(iteration, nodes_quantity):
            if not dfs[i]:
                for j in range(nodes_quantity):
                    if matrix[i][j] == 1:
                        print("Graf niespojny")
                        return False

    if oddnodes == 0:
        print("graf zawiera cykl eulera")
        return True
    else:
        return False

# wynik
reseuler = []


def eulerDFS(matrix, node):
    global reseuler

    nodesquan = len(matrix)
    for i in range(nodesquan):
        if matrix[node][i] == 1:
            matrix[node][i] = 0
            matrix[i][node] = 0
            eulerDFS(matrix, i)
    reseuler.append(node)


def euler_search(matrix):
    global reseuler
    reseuler = []

    eulerDFS(matrix, 0)

    res = []
    for i in reseuler:
        res.append(i)
    return res

# dla grafu skierowanego
cvn = 0
VN = []
VLow = []
VS = []
Vind = []
Voutd = []
C = []
S = []
stack = []
global m
global l

def dfsscc(v, graf):
    global cvn
    global VN
    global VLow
    global VS
    global Vind
    global Voutd
    global C
    global S
    cvn += 1
    VN[v] = cvn - 1
    VLow[v] = cvn - 1
    S.append(v)
    VS[v] = True
    for u in graf[v]:
        Voutd[v] += 1
        Vind[u] += 1
        if VN[u] == -1:
            dfsscc(u, graf)
            VLow[v] = min(VLow[v], VLow[u])
            break
        if VS[u] == False:
            break
        else:
            VLow[v] = min(VLow[v], VN[u])


    if VLow[v] != VN[v]:
        return

    u = S.pop()
    VS[u] = False
    C[u] = v
    if u != v:
        while u != v:
            u = S.pop()
            VS[u] = False
            C[u] = v
    return

# n to liczba wierzcholkow w grafie
def isEuleerian(n,graf):
    global cvn
    global VN
    global VLow
    global VS
    global Vind
    global Voutd
    global C
    global S

    VN = []
    VLow = []
    VS = []
    Vind = []
    Voutd = []
    C = []
    cvn = 0
    S = []

    # zerowanie tablic
    for i in range(len(graf)):
        VN.append(-1)

    for i in range(len(graf)):
        C.append(-1)

    for i in range(len(graf)):
        VS.append(False)

    for i in range(len(graf)):
        Vind.append(0)

    for i in range(len(graf)):
        Voutd.append(0)

    for i in range(len(graf)):
        VLow.append(-1)

    for v in range(len(graf)):
        if VN[v] == -1:
            dfsscc(v, graf)

    v = 0
    while (v < n) and (Vind[v] + Voutd[v]) == 0:
        v += 1
    if v == n:
        return 0

    cvn = C[v]
    cinout = 0
    coutin = 0
    print(C[v])
    while v < n:
        #print(v)
        if Vind[v] + Voutd[v] == 0:
            v += 1
            break
        if C[v] != cvn:
            return 0
        if Vind[v] == Voutd[v]:
            v += 1
            break
        if Vind[v] - Voutd[v] == 1:
            cinout += 1
            if cinout > 1:
                return 0
            v += 1
        if Voutd[v] - Vind[v] == 1:
            coutin += 1
            if coutin > 1:
                return 0
            v += 1
        if Voutd[v] - Vind[v] > 1 or Vind[v] - Voutd[v] > 1:
            return 0
    if cinout == 1:
        return 1
    else:
        return 2


def DFS_DiEuler(v):
    global stack
    global m
    while len(m[v]) > 0:
        tmp = m[v][0]
        m[v] = m[v][1:]
        DFS_DiEuler(tmp)
    stack.append(v)


def searchForDiEuler(list):
    if isEuleerian(len(list), list) != 2:
        return False

    global m
    global stack
    m = list

    #wybr pierwszego wierzcholka z niezerowa liczba sasiadow
    v = 0
    for i in range(len(m)):
        if len(m[i]) > 0:
            v = i
            break

    DFS_DiEuler(v)
    result = []
    result = stack
    m = []
    stack = []
    return result[-1::-1]


def testy():
    ############# Wywolania #####################
    quantities = [15,16,17,18,19,20,21,22,23,24,25]
    saturation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # testy dla grafow nieskierowanych hamilton
    '''
    plik = open('wyniki_nieskierowany_hamilton.txt', 'a')
    for i in saturation:
        print(i)
        plik.write(str(i) + '\n')
        for j in quantities:
            graph = createRandomGraph(j, i)
            savetofile('nieskierowany' + str(i) + '.txt', graph)
            matrix = create_neighbourhood_matrix('nieskierowany' + str(i) + '.txt')
            start = time.time()
            algorithm = rf_algorithm_nontargeted(matrix)
            stop = time.time()
            res = stop - start
            plik.write(str(j) + ' ' + str(res) + '\n')
        plik.write('\n')
    plik.close()
    '''
    # testy dla grafow nieskierowanych euler
    '''
    plik = open('wyniki_nieskierowany_euler.txt', 'a')
    for i in saturation:
        print(i)
        plik.write(str(i) + '\n')
        for j in quantities:
            graph = createRandomGraph(j, i)
            savetofile('nieskierowany' + str(i) + '.txt', graph)
            matrix = create_neighbourhood_matrix('nieskierowany' + str(i) + '.txt')
            start = time.time()
            if possible(matrix):
                algorithm = euler_search(matrix)
            stop = time.time()
            res = stop - start
            plik.write(str(j) + ' ' + str(res) + '\n')
        plik.write('\n')
    plik.close()
    '''

    # testy dla grafow skierowanych hamilton
    '''
    plik = open('wyniki_skierowany_hamilton.txt', 'a')
    for i in saturation:
        print(i)
        plik.write(str(i) + '\n')
        for j in quantities:
            graph = createrandomDiGraph(j, 'skierowany' + str(i) + '.txt',  i)
            lista = create_nextto('skierowany' + str(i) + '.txt')
            start = time.time()
            algorithm = rf_algorithm_targeted(lista)
            stop = time.time()
            res = stop - start
            plik.write(str(j) + ' ' + str(res) + '\n')
        plik.write('\n')
    plik.close()
    '''

    # testy dla grafow skierowanych euler
    plik = open('wyniki_skierowany_euler.txt', 'a')
    for i in saturation:
        print(i)
        plik.write(str(i) + '\n')
        for j in quantities:
            graph = createrandomDiGraph(j, 'skierowany' + str(i) + '.txt', i)
            lista = create_nextto('skierowany' + str(i) + '.txt')
            start = time.time()
            searchForDiEuler(lista)
            stop = time.time()
            res = stop - start
            plik.write(str(j) + ' ' + str(res) + '\n')
        plik.write('\n')
    plik.close()

    #sasiedztwaeuler = create_neighbourhood_matrix('nieskierowany.txt')
    #iseuler = possible(sasiedztwaeuler)
    #if iseuler:
       # print('cykl eulera dla grafu nieskierowanego')
       # res = euler_search(sasiedztwaeuler)
       # print(res)

x = threading.Thread(target=testy())
x.start()
