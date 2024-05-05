import os
import _pickle as pickle
import datetime
import pathos.multiprocessing as multiprocessing
from pathos.pools import ProcessPool
import pathos.profile as pr
import time

# Number of wanted simulations
sims_to_get = int(100)


##### CHECKS #####
# Check if the folder "SummaryStatistics" exists
assert os.path.exists("SummaryStatistics"), "The folder 'SummaryStatistics' does not exist."

# Check if the file "SummaryStatistics/done.pkl" exists
assert os.path.exists("SummaryStatistics/done.pkl"), "The file 'SummaryStatistics/done.pkl' does not exist."

# Load the file "SummaryStatistics/done.pkl"
with open("SummaryStatistics/done.pkl", "rb") as f:
    done = pickle.load(f)

# Check if the variable "done" is a dictionary
assert isinstance(done, dict), "File 'done.pkl' is not a dictionary."

# Get the last update of the file "done.pkl"
last_update = done["LastUpdate"]
print("The last update of 'done.pkl' was on " + last_update + ".")

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



##### PROCESS #####
processed_simulations = 0
processed_files = 0
processed_folders = 0

while(processed_simulations < sims_to_get and processed_folders < len(folders_in_simulation)):
    # Get the folder name and the files in the folder from Simulations
    folder_to_analyze = folders_in_simulation[processed_folders]
    files_to_analyze = os.listdir(os.path.join("Simulations", folder_to_analyze))

    # Check if there are new files to analyze
    file_to_analyze = [f for f in files_to_analyze if f.endswith(".pkl") and f not in done["ProcessedFiles"]]
    if len(files_to_analyze) > 0:
        # Check if the same folder is in SummaryStatistics
        if folder_to_analyze not in folders_in_summary_statistics:
            # Create the folder in SummaryStatistics
            os.mkdir(os.path.join("SummaryStatistics", folder_to_analyze))
            print("The folder " + folder_to_analyze + " has been created in 'SummaryStatistics'.")
print("Completed the processing of " + str(processed_files) + " files over " + str(processed_folders) + " folders.")





def GetSummaryStatisticsParallel(file_to_analyze, cores=-1):
    if cores == -1:
        cores = multiprocessing.cpu_count()
    if cores > multiprocessing.cpu_count():
        print(f"WARNING: You are using {cores} cores, but you have only {multiprocessing.cpu_count()} cores available")
    pool = ProcessPool(nodes=cores)

    start = time.time()
    with ProcessPool(nodes=cores) as pool:
        pool.map(AnalyzeFile, file_to_analyze)
    end = time.time()
    last_update = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Open the text file in write mode
    with open("SummaryStatistics/done.txt", "a") as file:
        file.write(last_update + "\n")
        for item in file_to_analyze: file.write(item + "\n")  # Add a newline after each item

    print(f"Parallel calculations took {end-start} seconds")



def AnalyzeFile(file):
    # Load the file
    with open(os.path.join("Simulations", folder_to_analyze, file), "rb") as f:
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
    summary_statistics = {"x0": x_trace[0]}

    # Save the summary statistics
    with open(os.path.join("SummaryStatistics", folder_to_analyze, "s"+file), "wb") as f:
        pickle.dump(summary_statistics, f)
    print("The summary statistics of " + file + " has been saved.")
