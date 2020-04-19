import csv

def create_file(path,file_name):
    with open(path+file_name, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    return

def add_row(path,file_name,col1,col2,col3):
    with open(path+file_name, mode='w',newline='') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([col1,col2,col3])
    return