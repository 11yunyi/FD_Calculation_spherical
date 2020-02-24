import csv


"""
Class for handling csv-files
"""


class _CSV:
    def __init__(self, filepath):
        # file handle
        self.csvfile = open(filepath, 'rb')
        # active flag
        self.flag_active = True

    # reader-functionality
    def reader(self):
        if self.flag_active:
            return csv.reader(self.csvfile, delimiter=';')
        else:
            print ("No active file")
            raise Exception('exit')

    # function to close the file-handle
    def close(self):
        # close the file handle
        self.csvfile.close()
        # set the flag to false
        self.flag_active = False
