# how to built a regression tree

"""整体思路：
1.root node
2.recursive split
3.terminal node (为了解决over-fitting的问题，减少整个tree的深度/高度，以及必须规定最小切分单位)
4.finish building the tree
"""

# 1
def dataset_test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

# 2
def calculate_the_gini_index(groups, classes):
    # 计算有多少实例
    n_instances = float(sum([len(group) for group in groups]))
    # 把每一个group里面的加权gini计算出来
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # *注意，这里不能除以0，所以我们要考虑到分母为0的情况
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # 这个做了一个加权处理
        gini += (1 - score) * (size / n_instances)

    return gini

# 3
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = 888, 888, 888, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = dataset_test_split(index, row[index], dataset)
            gini = calculate_the_gini_index(groups, class_values)
            # print("X%d < %.3f Gini=%.3f" % ((index + 1), row[index], gini))
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}

# 4 返回结果值中数量最多的那一项（如示例中：0或1）
def determine_the_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# root进一步分裂形成node和terminal的思路：
# 1.把数据进行切分（分为左边与右边），原数据删除掉
# 2.检查非空以及满足我们的我们设置的条件（深度/最小切分单位/非空）
# 3.一直重复类似寻找root node的操作，一直到最末端

# 5  注意：node变量是一个字典。get_split返回字典里的groups里是两个列表。
def split(node, max_depth, min_size, depth):
    # 做切分，并删除掉原数据
    # get_split返回字典里的groups里是两个列表。此处，left和right赋值后，是两个列表
    left, right = node['groups']
    # print("left",left)
    # print("right",right)
    # del后，node字典里就没有groups项了,只有index和value两项。
    del (node['groups'])
    # print(node)
    # 查看非空
    # 判断left为空或right为空
    if not left or not right:
        # left或right中有一项为空，则确定终点，终点值为
        node['left'] = node['right'] = determine_the_terminal(left + right)
        # print(node)
        return

    # 检查最大深度是否超过
    if depth >= max_depth:
        node['left'], node['right'] = determine_the_terminal(left), determine_the_terminal(right)
        # print(node)
        return
        # 最小分类判断与左侧继续向下分类
    if len(left) <= min_size:
        node['left'] = determine_the_terminal(left)
    else:
        node['left'] = get_split(left)
        # 递推函数split（），触发条件后，反复调用自己。
        split(node['left'], max_depth, min_size, depth + 1)
        # 最小分类判断与右侧继续向下分类
    if len(right) <= min_size:
        node['right'] = determine_the_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# 5 核心目的
# 最终建立决策树
def build_the_regression_tree(train, max_depth, min_size):
    root = get_split(train)
    # print(root)
    split(root, max_depth, min_size, 1)
    # print(root)
    return root

# 6
# 通过CLI可视化的呈现类树状结构便于感性认知
def print_our_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * '-', (node['index'] + 1), node['value'])))
        print_our_tree(node['left'], depth + 1)
        print_our_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '-', node)))

# 7
def make_prediction(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return make_prediction(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return make_prediction(node['right'], row)
        else:
            return node['right']


dataset = [[2.1, 1.1, 0],
           [3.4, 2.5, 0],
           [1.3, 5.8, 0],
           [1.9, 8.6, 0],
           [3.7, 6.2, 0],
           [8.8, 1.1, 1],
           [9.6, 3.4, 1],
           [10.2, 7.4, 1],
           [7.7, 8.8, 1],
           [9.7, 6.9, 1]]

tree = build_the_regression_tree(dataset, 3, 1)
print_our_tree(tree)

decision_tree_stump = {'index': 0, 'right': 1, 'value': 9.3, 'left': 0}
# for row in dataset:
#     prediction = make_prediction(decision_tree_stump, row)
#     print("What is expected data : %d , Your prediction is %d " % (row[-1], prediction))

"""实际输出结果与整体思路
输出结果：
{
'index': 0, 
'value': 7.7, 
'left': {'index': 0,
         'value': 2.1,
         'left': {'index': 0,
                  'value': 1.3, 
                  'left': 0, 
                  'right': 0}, 
         'right': {'index': 0, 
                   'value': 2.1,
                   'left': 0, 
                   'right': 0}},
'right': {'index': 0,
          'value': 8.8, 
          'left': 1, 
          'right': {'index': 0, 
                    'value': 8.8, 
                    'left': 1, 
                    'right': 1}}
}

思路：
实质是计算root并不断更新root，root是一个字典，由index、value、groups三部分组成，其中groups分为left和right。
返回值root是root根、node节点、terminal终点三类细分的统称。第一次对原数据的分裂（get split）得到root根，分为left和right。
然后对left和right进行判断，决定是继续分裂，形成node节点，还是终止分裂，形成terminal终点。

函数的运行顺序：
build_the_regression_tree()
    |
    ----> get_split() 循环遍历分裂方式，依据最优gini，确定分裂，返回index，value，left，right字典，赋值给root
    |           |
    |           ----> dataset_test_split() 将数据分裂为左右两部分，返回left和right
    |           |
    |           ----> calculate_the_gini_index() 对left和right分别计算gini系数，最终确定此分类gini系数
    |
    ----> split() 判断是否空集、判断是否达最大深度、判断是否继续分裂；如继续分裂，先分组，再执行split递推
            |
            ----> determine_the_terminal()
            |
            ----> get_split()
                        |
                        ----> split()
"""

