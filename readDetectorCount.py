
import csv

#currently return a list contains only second column (approximate count), can change if need
def readFromcsv() -> list[str]:
    with open("./strawberries.csv", "r") as file:
        content = csv.reader(file)
        return [x[1] for x in content]


if (__name__ == "__main__"):
    print(readFromcsv())
