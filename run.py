import copy
import datetime
import argparse
import sys

import help_functions as hf
import machine_learning as ml
import attribute_runner as ar
import real_time_simulation as rts

#Pipeline, die alle Methoden aufruft und Objekte erstellt

# Versuche die Einstellungen aus der Usereingabe zu erhalten
def handle_input(input):
    dic = {}
    ints = ['window', 'mini_window', 'repeats']

    # Alle moeglichen ML-Algorithmen und Attribute sind in den help_functions
    mlAlgos = hf.get_mlAlgos() + ['ml_optimized']
    attributes = ['pre_optimized_1', 'pre_optimized_2', 'all', 'optimize', 'manual_select']

    # Wenn das Stichwort "help" in der Eingabe ist gib eine Hilfestellung
    try:
        if 'help' in input:
            print('the first three given numbers set the size of the window,'
                  ' the mini-window and the repeats (in this order only)')
            print(f'the possible options for machine learning classification algorithms are: {mlAlgos}')
            print(f'the possible options for the attributes are: {attributes}')
            quit()
    except Exception:
        pass

    # Wenn keine Usereingabe erfolgt nutze die Standardeinstellung
    dic['mlAlgo'] = 'decision_trees'
    dic['attributes'] = 'pre_optimized'
    dic[ints[0]] = 1000
    dic[ints[1]] = 5
    dic[ints[2]] = 100

    if input is not None:
        # Behandle potenziell gegebene Floats
        input = hf.str_to_float(hf.replace_comma(input))

        j = 0
        for i in input:
            if type(i) is float and j < len(ints):
                dic[ints[j]] = int(i)
                j += 1
            elif type(i) is str:
                if i.lower() in mlAlgos:
                    dic['mlAlgo'] = i
                elif i.lower() in attributes:
                    dic['attributes'] = i

        # Wenn ungueltige Zahlen gegeben sind nutze stattdessen den Standard
        if dic['window'] < 1:
            dic['window'] = 1000
        if dic['mini_window'] < 1:
            dic['mini_window'] = 5
        if dic['repeats'] < 1:
            dic['repeats'] = 100

        # mini-window has to be smaller than window
        if dic['window'] < dic['mini_window']:
            dic['window'] = dic['mini_window'] + 1

    return dic

# Frage den Nutzer ob die Eingabe korrekt ist und der Prozess starten soll
def handle_proceed_input():
    proceed = input('proceed? [y/n] \n')
    if proceed.lower() in ['y', '1', 'yes']:
        return True
    elif proceed.lower() in ['n', '0', 'no']:
        quit()
    else:
        print('ERROR: invalid input')
        handle_proceed_input()

# Main des Prototyps
# Erstelle nacheinander alle Objekte und rufe die Methoden auf
def main():
    # Usereingabe untersuchen das Stichwort ist "--input"
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', action='store', type=str, nargs='*')

        args = parser.parse_args()

    except:
        e = sys.exc_info()[0]
        print(e)
        quit()

    input_dic = handle_input(args.input)
    print('The following settings where selected')
    for i in input_dic:
        print(f'{i}: {input_dic[i]}')
    handle_proceed_input()
    time_start = datetime.datetime.now()

    # Starte die ML-Klassifikation mit den gegebenen Optionen
    # Erstelle das Objekt, dass die Attribute bestimmt
    # Erstelle das Objekt, dass die Real-Time-Simulation fuer den Datensatz uebernimmt
    # Je mehr CSV-Dateien gegeben werden, desto mehr Objekte werden erstellt
    machine_learning = {}
    attribute_objects = {}
    real_time_objects = {}
    for case in hf.get_cases():
        machine_learning[case] = ml.Classification(case,
                                                   input_dic['mlAlgo'],
                                                   input_dic['repeats'],
                                                   input_dic['attributes'])
        machine_learning[case].start_learning()
        attribute_objects[case] = ar.AttributeRunner(case,
                                                     input_dic['window'],
                                                     input_dic['mini_window'],
                                                     machine_learning[case])
        real_time_objects[case] = rts.RealTimeSimulation(case)

    # Die Echtzeitsimulation wird die Daten Zeile fuer Zeile ausgeben
    # Wenn alle verfuegbaren eines Datensatzes ausgegeben wurden, gibt die Echtzeitsimulation None aus
    # Mit den ausscheidenen Datensaetzen verkleinert sich das Dictionary mit den Objekten
    while len(real_time_objects) > 0:
        keys = copy.deepcopy(list(real_time_objects.keys()))
        for nd in keys:
            new_row = real_time_objects[nd].get_new_data()
            if new_row is None:
                real_time_objects.pop(nd, None)
            else:
                attribute_objects[nd].runner(new_row)
    time_end = datetime.datetime.now()
    print(f'time needed: {time_end - time_start}')


if __name__ == '__main__':
    main()

