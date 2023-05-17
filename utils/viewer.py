import csv, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def merge_tables(base_path,folders_to_be_merged,methods,metric,out_name):
    is_build = False
    final_table = []
    for i in range(len(folders_to_be_merged)):
        with open(base_path + folders_to_be_merged[i] + "/" + metric + ".csv") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    line_count = 0
                    for row in csv_reader:
                        if i == 0 and is_build == False:
                            final_table = np.zeros( (len(methods),len(row)-1) )
                            is_build = True
                        if line_count > 0 and row[0] in methods:
                            for j in range(1,len(row)):
                                final_table[methods.index(row[0]),j-1] += float(row[j])
                        line_count += 1
    final_table = np.around(final_table/4,decimals=1)
    final_table = final_table.tolist()
    for i in range(len(final_table)):
        final_table[i].reverse()
        final_table[i].append(methods[i])
        final_table[i].reverse()
    
    out_file = base_path + out_name + "/" + metric + ".csv"
    with open(out_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(out_name+ ",Dir.,Disc.,Eat,Greet,Phone,Photo,Pose,Purch.,Sit,SitD.,Smoke,Wait,WalkD.,Walk,WalkT.,Avg.")
        writer.writerows(final_table)
    
    # LATEX
    latex = table2latex(final_table)
    out_file = base_path + out_name + "/" + metric + ".tex"
    with open(out_file, 'w') as f:
        for line in latex:
            f.write(line)
            f.write('\n')

def make_overview_table(base_path,errors,methods, metrics,out_name):
    final_table = np.zeros( (len(methods),len(metrics)*len(errors)) )
    # Populate table
    for i in range(len(errors)):    
        for j in range(len(metrics)):
            with open(base_path + errors[i] + "/" + metrics[j] + ".csv") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                for row in csv_reader:
                    if line_count > 0 and row[0] in methods:
                        final_table[methods.index(row[0]),len(metrics)*i+j] = float(row[-1])
                    line_count += 1
    final_table = final_table.tolist()
    for i in range(len(final_table)):
        final_table[i].reverse()
        final_table[i].append(methods[i])
        final_table[i].reverse()
    # LATEX
    latex = table2latex(final_table)
    out_file = base_path  + out_name + "_overview.tex"
    with open(out_file, 'w') as f:
        for line in latex:
            f.write(line)
            f.write('\n')

def table2latex(T):
    # Spot the lowest values
    t = T[1:]
    t = [row[1:] for row in t]
    t = np.array(t)
    best = [min(t[:,i]) for i in range(t.shape[1])]
    testo = []
    counter = 0

    # Intestazione
    testo.append(r"\begin{table*}[t]")
    testo.append(r"\centering")
    testo.append(r"\resizebox{\textwidth}{!}{%")
    testo.append(r"\begin{tabular}{l"+ "c"*t.shape[1] + "}")
    testo.append(r"\hline")

    for row in T:
        stringa = ""
        column_counter = 0
        for r in row:
            if counter > 0 and column_counter > 0 and r == best[column_counter-1]:
                stringa += r" & \textbf{" + str(r) + "}"
            else:
                if stringa == "":
                    stringa += str(r)
                else:
                    stringa += " & " + str(r)
            column_counter += 1
        testo.append(stringa + r"\\")
        if counter == 0:
            testo.append(r"\hline")
        counter += 1
    # Footer
    testo.append(r"\hline")
    testo.append(r"\end{tabular}%")
    testo.append(r"}")
    testo.append(r"\caption{TODO caption}")
    testo.append(r"\label{tab:my-table-TODO}")
    testo.append(r"\end{table*}")
    
    return testo

def table2md(T):
    # Spot the lowest values
    t = T[1:]
    t = [row[1:] for row in t]
    t = np.array(t)
    best = [min(t[:,i]) for i in range(t.shape[1])]
    testo = []
    counter = 0
    for row in T:
        stringa = "|"
        column_counter = 0
        for r in row:
            if counter > 0 and column_counter > 0 and r == best[column_counter-1]:
                stringa += "**" + str(r) + "**\t|"
            else:
                stringa += str(r) + "\t|"
            column_counter += 1
        testo.append(stringa)
        if counter == 0:
            stringa = "| "
            for r in row:
                stringa += "--- | "
            testo.append(stringa)
        counter += 1
    return testo

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

def write_time(tot,pre,run,post,kps,f,out_path):
     with open(out_path, mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["tot","pre","run","post","kps","frames"])
        writer.writerow([tot,pre,run,post,kps,f])

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