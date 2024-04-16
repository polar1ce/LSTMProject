# LSTM 小车身份认证

## 1. 定义行为模式

正常的小车的行为模式和错误小车的行为模式

## 2. 提升性能

* 增加属性数量
  
目前的属性是：小车的速度、直线行驶、电量消耗、运行时间 ${x_{t}, y_{t}, z_{t}}$，属性个数是`INPUT_SIZE = 3`

* 增加LSTM层数

当前是

```python
HIDDEN_SIZE = 10 
NUM_LAYERS = 1
```

* 对比试验

## 3. 测试时延

```python
import time
start_time = time.time()  # 记录开始时间
some_function()  # 调用函数
end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算运行时间
print(f"函数运行时间: {execution_time}秒")
```

## 4. 全过程模拟
