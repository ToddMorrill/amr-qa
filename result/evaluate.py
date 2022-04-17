import sys
import json


#Calculate precision, recall and F-measure per question and averaged the values.

system_result = sys.argv[1]
standard_result = sys.argv[2]

with open(standard_result, 'r') as f:
	standard = json.load(f)

with open(system_result, 'r') as file:
	system = json.load(file)

# Macro F-measure
# Measure percision, recall and F-measure per question and average the value.
recall = 0
precision = 0
f = 0

for result in system:
	try:
		index = result["id"]
		# ! Format undecided yet
		results = result["result"]
		#resultset =  result["results"]["bindings"]
		#results = [x.popitem()[1]["value"] for x in resultset]
	except:
		print("Missing result or format doesn't match")
		#results = [result["boolean"]]
		#print(result)
	answers = []
	for ans in standard:
		if ans["id"] == index:
			try:
				answerset = ans["results"]["bindings"]
				answers = [x.popitem()[1]["value"] for x in answerset]
				# Example: ['http://dbpedia.org/resource/Gulf_War', 'http://dbpedia.org/resource/Lebanese_Civil_War', 'http://dbpedia.org/resource/Yom_Kippur_War']
				# Example: ['http://dbpedia.org/resource/United_States_Senate', 'Governor of Texas', 'from Texas']
			except:
				answers = [ans["boolean"]]
			break

	# 1. Measure for each question
	# Recall = (# correct ans) / (# standard  ans)
	# Precision = (# correct ans) / (# system ans)
	# F-measure = (2 * Precision * Recall) / (Precision + Recall)
	
	if len(answers) == 0:
		if len(results) == 0:
			recall += 1
			precision += 1
			f += 1
	else:
		if len(results) != 0:
			correct = 0
			for r in results:
				if r in answers:
					correct += 1
			rq = correct/len(answers)
			pq = correct/len(results)
			try:
				f += 2*rq*pq/(pq+rq)
			except:
				f += 0
			recall += rq
			precision += pq

# 2. Calcuate average
final_recall = recall/len(system)
final_precision = precision/len(system)
final_f = f/len(system)
print("The Recall of the system is: ", "{:.2}".format(final_recall))
print("The Precision of the system is: ", "{:.2}".format(final_precision))
print("The F-measure of the system is: ", "{:.2}".format(final_f))






