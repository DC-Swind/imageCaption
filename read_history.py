import cPickle
history_path = "./log_st/setting14/histories__attention.pkl"
#history_path = "./log_st/histories__attention.pkl"

with open(history_path, 'rb') as f:
    infos = cPickle.load(f)
    val_history = infos['val_result_history']
    print len(val_history)
    for i in range(84):
        try:
            print i, val_history[(i+1)*2500]['loss']
        except:
            print ""

    print history_path



