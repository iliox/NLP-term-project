import pickle

def convert(filename):
    text=""
    with open(filename,"r") as F:
        for line in F:
            #print(line.split("\t")[0],line.split("\t")[1][:-1])
            #exit(0)
            if len(line)>1:
            	text = text + line.split("\t")[0]+ "\t" + "NN" + "\t" + "O" + "\t" + line.split("\t")[1]
            if len(line)<=1:
                text = text + "\n"
    with open("conll_"+filename, "w") as text_file:
    	text_file.write(text)

convert("train.txt")
convert("test.txt")
convert("dev.txt")
