import torch


def cfsgm(x, y, net, criterion, device, alpha=20):
    x = x.detach().clone().to(device)
    y = y.to(device)

    x.requires_grad = True

    outputs = net(x)
    cost = criterion(outputs, y)

    grad = torch.autograd.grad(
        cost, x,
        retain_graph=False,
        create_graph=False)[0]

    adv_images = x - alpha * grad

    return adv_images, y, grad

