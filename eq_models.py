import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from equivariant_layers_expand import eops_1_to_1, eops_1_to_2, eops_2_to_1, eops_2_to_2, eset_ops_3_to_3, eset_ops_4_to_4, eset_ops_1_to_3

class Eq1to1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq1to1, self).__init__()
        self.basis_dim = 2
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1))

    def forward(self, inputs):
        ops = eops_1_to_1(inputs)
        output = torch.einsum('dsb, nibd->nis', self.coefs, ops)
        output = output + self.bias
        return output

class Eq2to1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq2to1, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1))

    def forward(self, inputs):
        '''
        inputs: N x D x m x m
        Returns: N x D x m
        '''
        ops = ops_2_to_1(inputs)
        output = torch.einsum('dsb,ndbi->nsi', self.coefs, ops)
        output = output + self.bias
        return output

class Eq1to2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq1to2, self).__init__()
        self.basis_dim = 5
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim)))
        # diag bias, all bias, mat diag bias
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(self, inputs):
        ops = eops_1_to_2(inputs)
        output = torch.einsum('dsb,ndbij->nsij', self.coefs, ops)
        output = output + self.bias
        return output

class Eq2to2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq2to2, self).__init__()
        self.basis_dim = 15
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2. / (in_dim + out_dim + self.basis_dim)), (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))
        self.diag_bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1))
        self.diag_eyes = {}
        self.diag_eye = None #torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, inputs):
        ops = eops_2_to_2(inputs)
        output = torch.einsum('dsb,ndbij->nsij', self.coefs, ops)

        n = output.shape[-1]
        if n not in self.diag_eyes:
            device = self.diag_bias.device
            diag_eye = torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)
            diag_eye = torch.eye(n).unsqueeze(0).unsqueeze(0).to(device)
            self.diag_eyes[n] = diag_eye

        diag_eye = self.diag_eyes[n]
        diag_bias = diag_eye.multiply(self.diag_bias)
        output = output + self.bias + diag_bias
        return output

class Eq1to2Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, layer_dims):
        self.layers = nn.ModuleList(
            [Eq1to2(in_dim, out_dim)] + \
            [Eq2to2(out_dim, out_dim) for _ in range(1, len(layer_dims))]
        )
        self.fc_out = nn.Linear(hid_dim, out_dim)

    def forward(self, inputs):
        '''
        inputs:  N x D x m
        outputs: N x D x m x m
        '''
        output = inputs
        for l in self.layers:
            inputs = F.relu(l(outputs))
        output = self.fc_out(output)
        return output

class Eq1to3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Eq1to3, self).__init__()
        self.basis_dim = 4
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2.0 / (in_dim + out_dim + self.basis_dim)),
                                  (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1, 1))

    def forward(self, x):
        ops = eset_ops_1_to_3(x)
        output = torch.einsum('dsb,ndbijk->nsijk', self.coefs, ops) # in/out/basis, batch/in/basis/ijk
        output = output + self.bias
        return output

class SetEq3to3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SetEq3to3, self).__init__()
        self.basis_dim = 19
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2.0 / (in_dim + out_dim + self.basis_dim)),
                                  (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1, 1))

    def forward(self, x):
        ops = eset_ops_3_to_3(x)
        output = torch.einsum('dsb,ndbijk->nsijk', self.coefs, ops) # in/out/basis, batch/in/basis/ijk
        output = output + self.bias
        return output

class SetEq4to4(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SetEq4to4, self).__init__()
        self.basis_dim = 69
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.coefs = nn.Parameter(torch.normal(0, np.sqrt(2.0 / (in_dim + out_dim + self.basis_dim)),
                                  (in_dim, out_dim, self.basis_dim)))
        self.bias = nn.Parameter(torch.zeros(1, out_dim, 1, 1, 1, 1))

    def forward(self, x):
        ops = eset_ops_4_to_4(x)
        output = torch.einsum('dsb,ndbijkl->nsijkl', self.coefs, ops) # in/out/basis, batch/in/basis/ijk
        output = output + self.bias
        return output

if __name__ == '__main__':
    N = 10
    d_in = 5
    d_hid = 3
    d_out = 1
    m = 3
    x4 = torch.rand(N, d_in, m, m, m, m)
    x3 = torch.rand(N, d_in, m, m, m)
    x2 = torch.rand(N, d_in, m, m)
    x1 = torch.rand(N, d_in, m)

    m12a = Eq1to2(d_in, d_hid)
    m12b = Eq1to2(d_hid, d_out)

    m21a = Eq2to1(d_in, d_hid)
    m21b = Eq2to1(d_hid, d_out)

    out_dim = 4
    layers = [(d_in, 6), (6, 3), (3, out_dim)]
    net = Net2to2(layers, out_dim)
    print(m12b(m21a(x2)).shape, f'expect {N} x 1 x {m} x {m}')
    print(m21b(m12a(x1)).shape, f'expect {N} x 1 x {m}')
    print(net(x2).shape, 'expect dim:', f'{N} x {m} x {m} x {out_dim}')

    n11 = Net1to1(layers, out_dim)
    x11 = torch.rand(N, m, d_in)
    print(n11(x11), '1->1')
    print('done 11')
    m33 = SetEq3to3(d_in, d_hid)
    print(m33(x3).shape, f'expect {N} x {d_hid} x {m} x {m} x {m}')
    m44 = SetEq4to4(d_in, d_hid)
    print(m44(x4).shape, f'expect {N} x {d_hid} x {m} x {m} x {m}, {m}')
