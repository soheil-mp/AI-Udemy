
# Making the Accuracy, Precision, Recall, F1 Score
TN, FN, FP, TP = cm[0][0], cm[1][0], cm[0][1], cm[1][1]
accuracy = (TP + TN)/(TP + TN + FP + FN)
precision = TP / (TP + FP)
recall =  TP / (TP + FN)
F1_score = (2 * precision * recall) / (precision + recall)

d = {'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1_score':F1_score}

pd.DataFrame(d,index=['Values'])
