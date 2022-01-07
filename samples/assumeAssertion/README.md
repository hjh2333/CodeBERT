预测token的形式预测assert语句statement
1、测试代码特性识别不出来，assertEquals分词为assert，Equ，als
2、tokennize 加Ġ（注意是字母G上加一点）是什么意思
2、预测statement时，如果mask符号效果会大打折扣
origin_code: assertEquals(expectedProduct, actualProduct);

mask1: assertEquals(<mask><mask><mask> <mask><mask><mask>
answer1:
expected, actual,  expected, needed, required,

Product, Products, Production,  product, Produ,

,, ,, (),, ",, .,,

 actual,  expected, actual,  current,  unexpected,

Product, Products, Production,  product,  Product,

mask2: assertEquals(<mask><mask><mask> <mask><mask>);
answer2:
expected, actual,  expected, needed, current,

Product, Products,  product, Production, ,,

 actual,  expected, ,, actual, ",,

Product,  product, Products, Production, Package,

);, Product, ");, ));, ),

 //,  },        , //,  ;,



raw_train.jsonl现在是12个测试方法名带empty的用例

jsonl文件中每行
0 id
1 测试方法名
2 测试方法体
3 测试上下文
4 focal方法体
5 生产方法上下文
6 assert