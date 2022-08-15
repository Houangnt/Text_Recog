import torch


def concat_ltr_rtl(ltr, rtl):
    """
    :param ltr: left-to-right targets
    :param rtl: right-to-left targets
    :return: concatenation
    """
    if rtl is None:
        return ltr
    else:
        return torch.cat((ltr, rtl), 0)
