import torch
from torch import nn

class DCGLossSimple(nn.Module):
    """
    DCG Loss implementation by Jerry. (2022.11.11)
    """
    def __init__(self, model):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model  = model.to(self.device)

    def _set_up(self, x, y):
        """
        Set up variables.
        """
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.x.requires_grad = True
        self.x.retain_grad()
        self.b = self.x.shape[0]

    def get_loss(self):
        """
        Get loss.
        """
        self.logits     = self.model(self.x)
        self.logp       = torch.log(torch.exp(self.logits.gather(1, self.y.unsqueeze(1))).squeeze() / torch.sum(torch.exp(self.logits), dim=1)) 
        self.grad1      = torch.autograd.grad(self.logp, self.x, torch.ones_like(self.logp), retain_graph=True, create_graph=True)[0]
        self.grad1      = self.grad1.view(self.b, -1)
        # self.loss1      = torch.mean(self.grad1.norm(p=2, dim=1)**2)
        self.loss1      = torch.mean(self.grad1.norm(p=2, dim=1))
        return self.loss1

    def forward(self, x, y):
        self._set_up(x, y)
        return self.get_loss()


class DCGLoss(nn.Module):
    """
    DCG Loss implementation by Jerry. (2022.11.11)
    """
    def __init__(self, model):
        super().__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model  = model.to(self.device)

    def _set_up(self, x, y):
        """
        Set up variables.
        """
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.x.requires_grad = True
        self.x.retain_grad()
        self.b = self.x.shape[0]

    def get_loss(self):
        """
        Get loss.
        """
        self.logits     = self.model(self.x)
        self.logp       = torch.log(torch.exp(self.logits.gather(1, self.y.unsqueeze(1))).squeeze() / torch.sum(torch.exp(self.logits), dim=1))
        self.grad1      = torch.autograd.grad(self.logp, self.x, torch.ones_like(self.logp), retain_graph=True, create_graph=True)[0]
        self.grad1      = self.grad1.view(self.b, -1)
        self.loss1      = torch.mean(self.grad1.norm(p=2, dim=1))
        self.x_vec      = self.x.view(self.b, -1)
        self.v          = torch.rand_like(self.x_vec)
        self.v          = self.v / torch.norm(self.v, dim=-1, keepdim=True)
        self.grad1_vec  = self.grad1
        self.grad1v_v   = torch.sum(self.v * self.grad1_vec, dim=1)
        self.grad2      = torch.autograd.grad(self.grad1v_v, self.x, torch.ones_like(self.grad1v_v), create_graph=True, retain_graph=True)[0]
        self.grad2_vec  = self.grad2.view(self.b, -1)
        self.grad2v_v   = torch.sum(self.v * self.grad2_vec, dim=1)
        self.loss2      = torch.mean(self.grad2v_v)
        self.loss       = self.loss1 + 2 * self.loss2
        return self.loss

    def forward(self, x, y):
        self._set_up(x, y)
        return self.get_loss()

class MyLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ce             = nn.CrossEntropyLoss()
        self.dcg            = DCGLossSimple(model)
        self.dcg            = DCGLoss(model)
        self.lambda_        = 6
    
    def forward(self, x, y_hat, y_gt):
        return self.ce(y_hat, y_gt) + self.lambda_ * self.dcg(x, y_gt)

