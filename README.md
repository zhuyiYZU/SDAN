评价指标

可以提前对每个用户的商品排好序！！！

zero_one_error：
对于每个用户，需要比较该用户的测试商品与该用户的所有商品的大小关系，两层循环。（测试商品的数量）*（该用户评分的所有商品的数量）。

AvrgPrecision：
对于每个用户，要对其预测向量进行排序（sort），然后对该预测矩阵遍历一次，算出每个召回率值处的准确率。

NDCG:
去每个用户推荐商品（预测得分）的topk，计算该topk排序前的dcg值a，排序后，计算dcg值b。ndcg = a/b。
