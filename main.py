# import featureVec
# import feactureVecWithLang

file_name = 'data/NER Hindi English Code Mixed Tweets.tsv'
rows=[]

f = open(file_name, "r")
f_lines = f.readlines()
f.close()

line_count = 0
for i in range(len(f_lines)):
  if f_lines[i] == "\n": 
    append_text = '""'
    line_count+=1
  else: 
    append_text = "sent: "+str(line_count)+"\t" 
  rows.append(append_text+f_lines[i])

with open('processed_data/annotatedVec.tsv', 'w') as f:
    for item in rows:
        f.write(item)

# featureVec.numericFeatures()
# feactureVecWithLang.numericFeatures()
