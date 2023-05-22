from os import system as sys
import toml

cutoffs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for s in cutoffs:
    dirname = "s_" + str(s).replace(".", "_")
    sys("cp -r ./merge ./" + dirname)
    data = toml.load(dirname + "/input.toml") 
    data['physics']['cutFraction'] = float(s)
    sys("cd " + dirname)
    sys("sbatch run.sh")
    sys("cd ..")
