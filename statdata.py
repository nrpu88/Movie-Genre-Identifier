import csv

ifile = csv.reader(open("romcom.csv","r"))

a = 0
b = 0
c = 0
d = 0
e = 0



for row in ifile:

	for i in row:
		
		if i == row[3]:
			if i == "Action":
				a = a + 1


			if i == "Comedy":
				b = b + 1
				


			if i == "Horror":
				c = c + 1
				


			if i == "Drama":
				d = d + 1
				


			if i == "Romance":
				e = e + 1
				

print("done")
print("Here are the stats:")
print("Action:"+str(a))
print("Comedy:"+str(b))
print("Thriller:"+str(c))
print("Drama:"+str(d))
print("Science Fiction:"+str(e))

