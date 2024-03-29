
# This file is generated by PaConvert ToolKit, please Don't edit it!
import paddle
import ipdb
def to(self, *args, **kwargs):
    args_list = ["x", "y", "non_blocking", "copy", "memory_format"]
    new_kwargs = {}
    for i, node in enumerate(args):
        k = args_list[i]
        new_kwargs[k] = node
    for node in kwargs:
        v = kwargs[node]
        new_kwargs[node] = v
    kwargs = new_kwargs
    # ipdb.set_trace()
    if not kwargs:
        return self
    elif "tensor" in kwargs:
        return paddle.cast(self, "{}.dtype".format(kwargs["tensor"]))
    elif "dtype" in kwargs:
        return paddle.cast(self, "{}".format(kwargs["dtype"]))
    elif "device" in kwargs and "dtype" not in kwargs:
        return self
    elif kwargs:
        if "y" not in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str) and kwargs["x"] not in ['cpu', 'cuda', 'ipu', 'xpu']:
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], paddle.Tensor):
                dtype = kwargs["x"].dtype
            else:
                dtype = self.dtype
            return paddle.cast(self, dtype)

        elif "y" in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str):
                if x not in ['cpu', 'cuda', 'ipu', 'xpu']:
                    dtype = kwargs["x"]
                else:
                    dtype = kwargs["y"] if isinstance(kwargs["y"], str) else self.dtype
            else:
                dtype = kwargs["x"]
            return paddle.cast(self, dtype)
        else:
            return self

setattr(paddle.Tensor, 'to', to)

def split(self, *args, **kwargs):
    if args:
        if len(args)==1:
            return paddle.split(self, self.shape[0]//args[0])
        else:
            return paddle.split(self, self.shape[args[1]]//args[0], args[1])
    elif kwargs:
        if  "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")
            kwargs["num_or_sections"] = self.shape[kwargs["axis"]]//kwargs.pop("split_size")
        else:
            kwargs["num_or_sections"] = self.shape[0]//kwargs.pop("split_size")
        return paddle.split(self, **kwargs)

setattr(paddle.Tensor, 'split', split)
