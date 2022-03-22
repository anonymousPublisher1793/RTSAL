import pandas as pd
# help_functions enthaelt benoetigte Methoden zur Datenaufbereitung
import help_functions as hf

# Erstellt ein Objekt, dass die unbehandelten Daten enthaelt
# Iteriert ueber die Daten und gibt sie zeilenweise aus
class RealTimeSimulation:

    def __init__(self, case):
        self.case = case
        self.data = self.get_data('data/', self.case)
        self.max_len = len(self.data['y'])
        self.index = 0

    # beziehe die Daten aus dem Data-Ordner
    def get_data(self, path, file):
        data = pd.read_csv(path + file, delimiter=';', header=0)
        y = hf.str_to_float(hf.replace_comma(data['y'].to_numpy()))
        x = hf.str_to_float(hf.replace_comma(data['x'].to_numpy()))

        data['y'] = y
        data['x'] = x

        # One of the columns had an ' ' in the end
        for column in data:
            if column[len(column) - 1] == ' ':
                new_column = column[:len(column) - 1]
                data = data.rename(columns={column: new_column})

        return data

    # Gib die aktuelle Zeile des Datensatzes aus
    def get_new_data(self):
        if self.index == self.max_len:
            return None
        else:
            row = self.data.iloc[self.index]

            if self.index % 1000 == 0:
                print(f'progress for {self.case}: {round( (self.index / self.max_len) * 100, 2)}%')

            self.index += 1
            return row.copy()
