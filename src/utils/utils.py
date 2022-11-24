import os

def find_best_ckpt(p):
    pths = [x for x in os.listdir(p) if x.endswith("pth")]
    pths.sort()
    return os.path.join(p, pths[-1])




def interactive_get_value(obj, key):
    if hasattr(obj, key):
        return getattr(obj, key)
    else:
        return input(f"请输入{key}的名称: ")


