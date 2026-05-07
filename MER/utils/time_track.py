import time
from functools import partial
from functools import wraps # 推荐加上 wraps，保留原函数的元信息

def base_time_desc_decorator(method, desc='test_description'):
    @wraps(method)
    def timed(*args, **kwargs):
        # 打印描述
        print(desc)

        # 记录开始时间
        start = time.time()

        # 直接运行方法，不要用 try-except 拦截异常！
        # 如果报错，就让它真实地暴露出来
        result = method(*args, **kwargs)

        # 打印耗时 (修复了格式化，建议用 {:.2f} 否则秒数太长可能显示科学计数法)
        print('Done! It took {:.2f} secs\n'.format(time.time() - start))

        return result

    return timed


def time_desc_decorator(desc): return partial(base_time_desc_decorator, desc=desc)


@time_desc_decorator('this is description')
def time_test(arg, kwarg='this is kwarg'):
    time.sleep(3)
    print('Inside of time_test')
    print('printing arg: ', arg)
    print('printing kwarg: ',  kwarg)


@time_desc_decorator('this is second description')
def no_arg_method():
    print('this method has no argument')


if __name__ == '__main__':
    time_test('hello', kwarg=3)
    time_test(3)
    no_arg_method()
