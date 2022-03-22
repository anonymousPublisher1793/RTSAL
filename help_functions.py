from os import listdir

# zentrale Sammlung aller Methoden, die in mehreren Klassen benoetigt werden

# Wird benoetigt um die Dezimalschreibweise der Daten von Komma zu Punkt anzupassen
def replace_comma(list):
    l = 0
    for i in list:
        if type(list[l]) is str:
            list[l] = list[l].replace(",", ".")

        l += 1
    return list

# Daten sind inital wegen der Dezimalkommata ein String, caste zu Float
def str_to_float(list):
    l = 0
    for i in list:
        try:
            list[l] = float(i)
        except Exception:
            pass
        l += 1
    return list

# Entferne Umlaute aus den gegebenen Labeln um die Lesbarkeit zu erhalten
def replace_umlauts(string):
    if type(string) is str:
        string = string.replace("ä", "ae")
        string = string.replace("Ä", "Ae")
        string = string.replace("ö", "oe")
        string = string.replace("Ö", "Oe")
        string = string.replace("ü", "ue")
        string = string.replace("Ü", "Ue")
        string = string.replace("ß", "ss")

    return string

# Gibt alle in einer Liste enthaltenen Label zurueck
def get_types(list):
    types = []
    for i in list:
        if i not in types:
            types += [i]
    return types

# Gibt ein Dictionary zurueck, dass alle Typen als Key und die groeße der Klassen als value enthaelt
def get_class_sizes(df, types):
    dic = {}
    for i in types:
        l = len(df.loc[(df['type'] == i)]['type'])
        dic[i] = l
    return dic

# Alle waehlbaren ML-Algorithmen
def get_mlAlgos():
    return ['decision_trees',
            'knn',
            'naive_bayes',
            'mlp_classifier']

# Gib alle Attribute die moeglich sind aus
def get_attributes():
    return ['Y',
            'delta_mavg_window',
            'mavg',
            'mini_std',
            'delta_amplitude',
            'slope',
            'is_group',
            'delta_E[Y]',
            'v(Y)']

# Optimale Attribute fuer Case 1
def get_pre_optimized_1():
    return ['Y',
            'delta_mavg_window',
            #'mavg',
            #'mini_std',
            'delta_amplitude',
            'slope',
            'is_group',
            'delta_E[Y]',
            #'v(Y)'
            ]

# Optimale Attribute fuer Case 2
def get_pre_optimized_2():
    return ['Y',
            'delta_mavg_window',
            'mavg',
            #'mini_std',
            #'delta_amplitude',
            'slope',
            'is_group',
            'delta_E[Y]',
            'v(Y)'
            ]

# Finde heraus, welche Dateien vom Nutzer bereitgestellt wurden
# Gibt die Namen der Dateien zurueck, diese werden im weiteren als Schluessel verwandt
def get_cases():
    # Dateien im Ordner mit den Learningdaten
    labelled_files = listdir('labelled_data')
    # Dateien die fuer Echtzeitsimulation verwandt werden sollen
    files = listdir('data')
    i = 0
    for f in labelled_files:
        # Es muss eine CSV-Datei sein und es muss eine korrespondierende Datei im Data-Ordner sein
        if '.csv' not in f or f not in files:
            labelled_files.pop(i)
        i += 1
    return labelled_files

# gibt zurueck mit welchen Schritten das Kontextattribut steigt
# Z.B. X= (1, 1.5, 2, 2.5) => 0.5
def get_x_steps(list):
    decimal_point = str(list[0]).index('.')
    decimal_places = len(str(list[0])[decimal_point + 1:])

    x_steps = list[1] - list[0]

    return round(x_steps, decimal_places), decimal_places
