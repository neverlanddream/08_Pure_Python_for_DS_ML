"""一些特殊用法、技巧的备注"""


# 关于用sum对列表求和降维的用法
sum_for_list_v1 = [[2 , 2 , 0],
                   [2 , 4 , 0],
                   [9 , 1 , 1],
                   [10, 4 , 1]]

sum_for_list_v2 = sum( sum_for_list_v1 , [] )
print(sum_for_list_v2)

#不和空列表求和会报错
# sum_for_list_v3 = sum( sum_for_list_v1 )
# print(sum_for_list_v3)