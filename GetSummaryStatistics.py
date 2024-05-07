import os
import _pickle as pickle
import datetime
import pathos.multiprocessing as multiprocessing
from pathos.pools import ProcessPool
import pathos.profile as pr
import time
from InternalLibrary.StatisticalFunctions import compute_summary_statistics
# Number of wanted simulations
sims_to_get = int(100)
max_files_to_analyze = 80



def GetSummaryStatisticsParallel(file_to_analyze, cores=-1):
    if cores == -1:
        cores = multiprocessing.cpu_count()
    if cores > multiprocessing.cpu_count():
        print(f"WARNING: You are using {cores} cores, but you have only {multiprocessing.cpu_count()} cores available")
    pool = ProcessPool(nodes=cores)
    print("Starting parallel calculations...")
    print("Processing " + str(len(file_to_analyze)) + " files.")
    start = time.time()
    with ProcessPool(nodes=cores) as pool:
        pool.map(AnalyzeFile, file_to_analyze)
    end = time.time()
    last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open the text file in write mode
    with open("SummaryStatistics/done.txt", "r") as file:
        file.readlines()
    
    lines[0]  = last_update + "\n"
    with open("SummaryStatistics/done.txt", "w") as file:
        file.writelines(lines)
        for item in file_to_analyze: file.write(item + "\n")  # Add a newline after each item

    print(f"Parallel calculations took {end-start} seconds")



def AnalyzeFile(file):
    # Load the file
    with open(os.path.join("Simulations", file), "rb") as f:
        data = pickle.load(f)
    print("The file " + file + " has been loaded.")

    # Check if the data is a dictionary
    assert isinstance(data, dict), "The pickle in " + file + " is not a dictionary."

    # Unpack the data
    x_trace = data["x_trace"]
    y_trace = data["y_trace"]
    f_trace = data["f_trace"]
    theta = data["theta"]
    n_sim = data["n_sim"]
    time_of_creation = data["time_of_creation"]

    # Compute the summary statistics (TO IMPLEMENT)
    summary_statistics = []
    for i in range(n_sim):
        s_stat = compute_summary_statistics(x_trace[i])
        summary_statistics.append(s_stat)

    # Save the summary statistics
    if not os.path.join("SummaryStatistics", file[:8]):
        os.mkdir(os.path.join("SummaryStatistics", file[:8]))
    with open(os.path.join("SummaryStatistics", file), "wb") as f:
        pickle.dump(summary_statistics, f)
    print("The summary statistics of " + file + " has been saved.")


##### CHECKS #####
# Check if the folder "SummaryStatistics" exists
assert os.path.exists("SummaryStatistics"), "The folder 'SummaryStatistics' does not exist."

# Check if the file "SummaryStatistics/done.txt" exists
assert os.path.exists("SummaryStatistics/done.txt"), "The file 'SummaryStatistics/done.txt' does not exist."

# Load the file "done.txt" and unpack the data
with open("SummaryStatistics/done.txt", "r") as file:
    lines = file.readlines()
last_update = lines[0].strip()
print("The last update of 'done.txt' was on " + last_update + ".")
done_list = [line.strip() for line in lines[1:]]

# List the elements of "SummaryStatistics" and get the folders
files_in_summary_statistics = os.listdir("SummaryStatistics")
folders_in_summary_statistics = [f for f in files_in_summary_statistics if os.path.isdir(os.path.join("SummaryStatistics", f))]

# Check if the folder "Simulation" exists
assert os.path.exists("Simulations"), "The folder 'Simulations' does not exist."

# List the elements of "Simulation" and check if it is not empty
files_in_simulation = os.listdir("Simulations")
assert len(files_in_simulation) > 0, "The folder 'Simulations' is empty."

# Get the folder names in "Simulation"
folders_in_simulation = [f for f in files_in_simulation if os.path.isdir(os.path.join("Simulations", f))]
assert len(folders_in_simulation) > 0, "There are no folders in 'Simulations'."
if len(files_in_simulation) > len(folders_in_simulation):
    print("Warning: There are files in 'Simulation' that are not folders.")


folders_inside_simulations = os.listdir("Simulations")
if ".DS_Store" in folders_inside_simulations:
    folders_inside_simulations.remove(".DS_Store")

# Create a list of all files inside "Simulations" with the folder name
files_in_simulation = []

for folder in folders_inside_simulations:
    temp = os.listdir(os.path.join("Simulations", folder))
    if ".DS_Store" in temp:
        temp.remove(".DS_Store")
    temp = [os.path.join(folder, f) for f in temp]
    files_in_simulation.extend(temp)

# Check if there are new files to analyze
file_to_analyze = [f for f in files_in_simulation if f.endswith(".pkl") and f not in done_list]
if len(file_to_analyze) > max_files_to_analyze: 
    file_to_analyze = file_to_analyze[:max_files_to_analyze]
print(file_to_analyze)


GetSummaryStatisticsParallel(file_to_analyze)



