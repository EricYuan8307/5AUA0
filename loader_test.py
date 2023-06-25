import torch

def get( index):
    items_input = []
    items_truth_class=[]
    items_truth_ins=[]
    items_truth_depth=[]
    if index%5 == 4: # 0 1 2 3 4 | 5 6 7 8 9| 10..    [4 5] [9 10] cannot be a pair!  这里有问题，跳过的话，return什么呢！
        return "SKIP" # dataloader will skip none
    else:
        for i in range(2):      
            input,truth_class,truth_ins,truth_depth = get_item_single(index+i)
            items_input.append(input)
            items_truth_class.append(truth_class)
            items_truth_ins.append(truth_ins)
            items_truth_depth.append(truth_depth)

        stacked_input=torch.stack(items_input,dim=0)    # run map if use stack,     each of the item ,  [2,3,H,W]
        stacked_truth_class= torch.stack(items_truth_class,dim=0) # [2,H,W]
        stacked_truth_ins= torch.stack(items_truth_ins,dim=0) # [2,H,W]
        stacked_truth_depth = torch.stack(items_truth_depth,dim=0) # [2,H,W]
    return stacked_input,stacked_truth_class,stacked_truth_ins,stacked_truth_depth

def get_item_single(index):
    a=torch.tensor([[1+index,2+index],[3,4]])
    b=torch.tensor([[11,12],[13,4]])
    c=torch.tensor([[21,22],[23,4]])
    d=torch.tensor([[31,32],[33,4]])
    return a,b,c,d

if __name__ == '__main__':
    for i in [1,2]:
        stacked_input,stacked_truth_class,stacked_truth_ins,stacked_truth_depth=get(i)
        print(stacked_input.shape)
        print(i)
        print('t')
        print(stacked_input[0])
        print('t next')
        print(stacked_input[1])
