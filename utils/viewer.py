import csv, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def write_table(output_path,table,time,names):
    with open(output_path, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(names)
        for i in range(1,len(time)):
            row = []
            row.append(time[i])
            for e in table[i]:
                row.append(e)
            writer.writerow(row)

# Input the path of a valid csv file, return the table
def get_table(input_path):
    table = []
    names = []
    time  = []
    with open(input_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                r = [float(number) for number in row]
                table.append(r[1:])
                time.append(r[0])
            else:
                names = row
            line_count +=1
    return table, time, names

# Plot the current variable over time
def view(tables,times,names,column_name):
    for i,T in enumerate(tables):
        if column_name in names[i]:
            id = names[i].index(column_name)-1
            col = [row[id] for row in T]
            plt.plot(times[i],col)
    plt.show()

# Parse argument if passed directly from viewer.py
def main():
    table1, time1, names1 = get_table(input_path=sys.argv[1])
    table2, time2, names2 = get_table(input_path=sys.argv[2])
    if len(sys.argv) > 4:
        table3, time3, names3 = get_table(input_path=sys.argv[3])
        view([table1,table2,table3],[time1,time2,time3],[names1,names2,names3],sys.argv[4])
    else:
        view([table1,table2],[time1,time2],[names1,names2],sys.argv[3])

if __name__ == "__main__":
    main()