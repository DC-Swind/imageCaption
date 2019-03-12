import json

pred1 = json.load(open("res/result_topdown_setting_11_ep20_prob.json", "r"))
print len(pred1)
print pred1[0].keys()
print pred1[0]['log_prob'], pred1[0]['caption']


pred2 = json.load(open("res/res_attention_wenhao_dense_for_ensemble.json", "r"))
print len(pred2)
print pred2[0].keys()


chose1 = 0
chose2 = 0
equal = 0
result = []
"""
for p1, p2 in zip(pred1, pred2):
    if p1["image_id"] != p2["image_id"]:
        print p1["image_id"]
        print p2["image_id"]
        exit(0)
    print p1["log_prob"], p1["caption"]
    print p2["prob"], p2["caption"]
    if p1["log_prob"] > p2["prob"]:
        chose1 += 1
    elif p1["log_prob"] < p2["prob"]:
        chose2 += 1
    else:
        equal += 1
"""
index = 0
for p1 in pred1:
    index += 1
    for p2 in pred2:
        if p1["image_id"] == p2["image_id"]:
            #print "--------------------------------------------"
            #print "index:", index, "image id:", p1["image_id"]
            #print "{:.5f}".format(p1["log_prob"]), p1["caption"]
            #print "{:.5f}".format(p2["prob"]), p2["caption"]

            res = {}
            res["image_id"] = p1["image_id"]

            if p1["log_prob"] > p2["prob"]:
                chose1 += 1
                res["caption"] = p1["caption"]
            elif p1["log_prob"] < p2["prob"]:
                chose2 += 1
                res["caption"] = p2["caption"]
            else:
                equal += 1
                res["caption"] = p2["caption"]
            result.append(res)
json.dump(result, open('./res/ensemble_result1.json', 'w+'))
print "chose1", chose1, "chose2", chose2, "equal", equal, "total", chose1 + chose2 + equal
