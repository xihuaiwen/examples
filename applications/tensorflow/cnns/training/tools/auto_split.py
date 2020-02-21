import tensorflow as tf

def build_graph_list(train_graph,ipu_num=None,name=None):
    #get node name
    mul_add = 1
    all_parameters = 0
    ops_name=[]
    ops_params=[]
    split_interval = 0
    split_result = []
    name = [{tensor.name:tensor.attr['shape']} for tensor in train_graph.as_graph_def().node if 'all/' in tensor.name]
    for i in name:
        for key,val in i.items():
            if len(val.shape.dim):
                #ops.append(i)
                mul_add = 1
                for j in range(len(val.shape.dim)):
                    mul_add *= val.shape.dim[j].size
                ops_name.append(key)
                ops_params.append(mul_add)
                all_parameters += mul_add
    
    each_split_params = all_parameters/ipu_num
    for l in range(ipu_num-1):
        split_interval = 0
        for k in range(len(ops_params)):
            split_interval += ops_params[k]
            if split_interval > each_split_params*(l+1):
                split_result.append(ops_name[k-1])
                break
        pass
    print("Parameters="+str(all_parameters))
    
    pass