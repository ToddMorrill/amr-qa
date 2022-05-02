import sys
import json


#Calculate precision, recall and F-measure per question and averaged the values.

system_result = sys.argv[1]
standard_result = sys.argv[2]

with open(standard_result, 'r') as f:
	standard = json.load(f)

with open(system_result, 'r') as f2:
	system = json.load(f2)

# Macro F-measure
# Measure percision, recall and F-measure per question and average the value.
# Macro F-measure
# Measure percision, recall and F-measure per question and average the value.
recall = 0
precision = 0
f = 0
fail = 0
empty = 0
both = 0
for result in system:
    index = result["id"]
    ix = int(index[6:])
    if result["status"] == "success":
		# ! Format undecided yet
        #results = result["results"]
        resultset = result["results"]["bindings"]
        results = [x.popitem()[1]["value"] for x in resultset]
        if len(results) == 0:
            empty += 1
    else:
        fail += 1
        results = {}
    answers = []
    #for ans in standard:
    ans = standard[ix]
    if ans["id"] == index:
            #print(index)
        try:
            answerset = ans["results"]["bindings"]
            answers = [x.popitem()[1]["value"] for x in answerset]
        except:
            answers = [ans["boolean"]]
    else:
        print("Wrong index:", ix, ans["id"])
        break

	# 1. Measure for each question
	# Recall = (# correct ans) / (# standard  ans)
	# Precision = (# correct ans) / (# system ans)
	# F-measure = (2 * Precision * Recall) / (Precision + Recall)
    
    if len(answers) == 0:
        if len(results) == 0:
            both += 1
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

final_recall = recall/len(system)
final_precision = precision/len(system)
final_f = f/len(system)
print("The Compared files are: ", system_result,',',  standard_result)
print("The Recall of the system is: ", "{:.2%}".format(final_recall))
print("The Precision of the system is: ", "{:.2%}".format(final_precision))
print("The F-measure of the system is: ", "{:.2%}".format(final_f))
print("The number of failed queries is: ", fail)
print("The number of succeeded queries is: ", len(system)-fail)
print("The number of empty generated result is", empty)
print("The number of empty result for both standard and generated is", both)

