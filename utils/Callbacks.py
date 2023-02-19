# -------------------------------
# -*- coding = utf-8 -*-
# @Time : 2023/2/12 13:40
# @Author : chc_stars
# @File : Callbacks.py
# @Software : PyCharm
# -------------------------------


class Callbacks:
    """
    处理YOLOv5挂钩的所有注册回调
    """
    # 定义可能的回调

    _callbacks = {
        'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [],

        'on_train_start': [],
        'on_train_epoch_start': [],
        'on_train_batch_start': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],

        'on_val_start': [],
        'on_val_batch_start': [],
        'on_val_image_end': [],
        'on_val_batch_end': [],
        'on_val_end': [],

        'on_fit_epoch_end': [],  # fit = train + val
        'on_model_save': [],
        'on_train_end': [],

        'teardown': [],
    }

    def register_action(self, hook, name='', callback=None):
        """
        注册一个新的行为来为回调挂钩
        :param hook:  要向其注册操作的回调钩子名称
        :param name:  操作的名称，供以后参考
        :param callback:  触发回调
        :return:
        """

        assert hook in self._callbacks, f"hook  '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """
        return all the registered actions by callback hook

        """
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        循环注册的动作并触发所有回调
        :param hook:
        :param args:
        :param kwargs:
        :return:
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)


if __name__ == '__main__':
    pass