from os import system
from os import chdir
import toml
import sys
import fileinput

cluster = "itp"
#cluster = "bw" 


cutoffs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for s in cutoffs:
    dirname = "s_" + str(s).replace(".", "_")

    # Create folde for this configuration
    system("mkdir " + dirname)

    # Copy files into new folder
    system("cp -r ./mistake/* " + dirname + "/")

    # Enter directory for this configuration
    chdir(dirname)

    # Load and modify toml params
    data = toml.load("./input.toml") 
    data['physics']['cutFraction'] = float(s)

    # Write new data to toml
    f = open("./input.toml",'w')
    toml.dump(data, f)
    f.close()

    # Submit job
    
    if cluster == "itp":
        filename = "runitp.sh"
        for line in fileinput.input(filename, inplace=1):
            if "cd" in line:
                line = line.replace("cd QuarkMesonModel/mistake", "cd QuarkMesonModel/" + dirname)
            sys.stdout.write(line)
        system("qsub runitp.sh")
    elif cluster == "bw":
        system("sbatch runbw.sh")

    # Return to parent directory
    chdir("../")
