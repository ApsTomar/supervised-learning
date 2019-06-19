import sys
import csv

# to extract some data from the dataset to create a small sample dataset.
def create_sample_dataset(num_of_lines):
    count = 0
    with open(inputFile, "r") as f, open(outputFile, "w") as out:
        reader = csv.reader(f)
        writer = csv.writer(out)
        for row in reader:
            writer.writerow(row)
            count += 1
            if count == num_of_lines:
                out.close()
                f.close()
                return out


inputFile = sys.argv[1]
outputFile = sys.argv[2]
numOfRows = sys.argv[3]
create_sample_dataset(numOfRows)
f_out = open(outputFile, "r")
print (f_out.read())
f_out.close()
