
import re
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    newpath = path.replace('.tsv', '.csv')
    # reading given tsv file

    with open(path, 'r', encoding="utf8") as myfile:

        with open(newpath, 'w',encoding="utf8") as csv_file:
            for line in myfile:
                # Replace every tab with comma
                line = line.replace(',', '')
                fileContent = re.sub("\t", ",", line)

                # Writing into csv file
                csv_file.write(fileContent)

    # output
    print("Successfully made csv file")