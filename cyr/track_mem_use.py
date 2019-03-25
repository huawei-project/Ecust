import datetime
import linecache
import os
import torch
import numpy as np
from py3nvml import py3nvml
import torch
import torch.nn as nn
from torch.autograd import Variable
import socket
import sys


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'

# different settings
print_tensor_sizes = False
use_incremental = False

if 'GPU_DEBUG' in os.environ:
    time = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d %H:%M:%S')
    gpu_profile_fn = "../../summary/Host_{}_gpu{}_mem_prof-{}.prof.txt". \
        format(socket.gethostname(), os.environ['GPU_DEBUG'], time)
    print('profiling gpu usage to ', gpu_profile_fn)


# Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None


def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global last_meminfo_used
    global lineno, func_name, filename, module_name

    if event == 'line':
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(
                    int(os.environ['GPU_DEBUG']))
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno)
                where_str = module_name + ' ' + func_name + ':' + str(lineno)

                new_meminfo_used = meminfo.used
                mem_display = new_meminfo_used - \
                    last_meminfo_used if use_incremental else new_meminfo_used
                with open(gpu_profile_fn, 'a+') as f:
                    f.write("{:<50}"
                            ":{:<7.1f}Mb "
                            "{}\n".format(where_str, (mem_display) / 1024**2, line.rstrip()))

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, 'dbg_alloc_where'):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {(type(x), tuple(x.size()), x.dbg_alloc_where)
                                            for x in get_tensors()}
                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(
                                '+ {:<50} {:<20} {:<10}\n'.format(loc, str(s), str(t)))
                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(
                                '- {:<50} {:<20} {:<10}\n'.format(loc, str(s), str(t)))
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if (filename.endswith(".pyc") or
                    filename.endswith(".pyo")):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            # only profile codes within the parenet folder, otherwise there are too many function calls into other pytorch scripts
            # need to modify the key words below to suit your case.
            if 'gpu_memory_profiling' not in os.path.dirname(os.path.abspath(filename)):
                lineno = None  # skip current line evaluation

            if ('car_datasets' in filename
                    or '_exec_config' in func_name
                    or 'gpu_profile' in module_name
                    or 'tee_stdout' in module_name):
                lineno = None  # skip othe unnecessary lines

            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass


def modelsize(model, input, type_size=4, Graphics=False):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(
        model._get_name(), para * type_size / 1000 / 1000))

    if Graphics:
        input_ = input.clone()
        input_.requires_grad_(requires_grad=False)
        use_cuda = input_.is_cuda

        # mods = list(model.modules())
        named_mods = list(model.named_modules())
        out_sizes = []

        i = 1
        while i < len(named_mods):
            n, m = named_mods[i]
            if isinstance(m, nn.ReLU):
                if m.inplace:
                    i += 1
                    continue
            if isinstance(m, nn.Sequential):
                if len(list(m.children())) == 2:
                    i += 3
                    continue
                else:
                    i += 1
                    continue
            if isinstance(m, Bottleneck):
                i += 1
                continue
            if isinstance(m, PixelShuffleBlock):
                i += 1

            if isinstance(m, nn.Conv2d):
                if n in ['aspp1.0', 'aspp2.0', 'aspp3.0', 'aspp4.0']:
                    input_ = Variable(torch.rand(input_.shape[0], 2048, 32, 44)).cuda() if use_cuda else \
                        Variable(torch.rand(input_.shape[0], 2048, 32, 44))
                    out = m(input_)
                else:
                    out = m(input_)
            if isinstance(m, nn.Dropout2d) and n == 'merge.0':
                input_ = Variable(torch.rand(input_.shape[0], 512 * 5, 32, 44)).cuda() if use_cuda else \
                    Variable(torch.rand(input_.shape[0], 512 * 5, 32, 44))
                out = m(input_)
            else:
                out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out
            i += 1

        total_nums = 0
        for i in range(len(out_sizes)):
            s = out_sizes[i]
            nums = np.prod(np.array(s))
            total_nums += nums

        print('Model {} : intermedite variables: {:3f} M (without backward)'
              .format(model._get_name(), total_nums * type_size / 1000 / 1000))
        print('Model {} : intermedite variables: {:3f} M (with backward)'
              .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))
