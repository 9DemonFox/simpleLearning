import itertools
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from data.rfanfis.dataLoader import ANFISDataLoader

dtype = torch.float


def calc_error(y_pred, y_actual):
    with torch.no_grad():
        tot_loss = F.mse_loss(y_pred, y_actual)
        rmse = torch.sqrt(tot_loss).item()
        perc_loss = torch.mean(100. * torch.abs((y_pred - y_actual)
                                                / y_actual))
    return (tot_loss, rmse, perc_loss)


def plotErrors(errors):
    '''
        Plot the given list of error rates against no. of epochs
    '''
    plt.plot(range(len(errors)), errors, '-ro', label='errors')
    plt.ylabel('Percentage error')
    plt.xlabel('Epoch')
    plt.show()


def train_anfis_with(model, data, optimizer, criterion,
                     epochs=500, show_plots=False):
    '''
        Train the given model using the given (x,y) data.
    '''
    # 处理data,将data从numpy二元组合并为dataLoader
    tensor_X = torch.from_numpy(data[0])
    tensor_y = torch.from_numpy(data[1])
    ds = TensorDataset(tensor_X, tensor_y)
    data = DataLoader(ds, batch_size=16, shuffle=True)

    errors = []  # Keep a list of these for plotting afterwards
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # print('### Training for {} epochs, training size = {} cases'.
    #      format(epochs, data.dataset.tensors[0].shape[0]))
    for t in range(epochs):
        # Process each mini-batch in turn:
        for x, y_actual in data:
            # print(x.size())
            y_pred = model(x)
            # Compute and print loss
            loss = criterion(y_pred, y_actual)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Epoch ending, so now fit the coefficients based on all data:
        x, y_actual = data.dataset.tensors
        with torch.no_grad():
            model.fit_coeff(x, y_actual)


def make_bell_mfs(a, b, clist):
    '''Return a list of bell mfs, same (a,b), list of centers'''
    return [BellMembFunc(a, b, c) for c in clist]


def _mk_param(val):
    '''Make a torch parameter from a scalar value'''
    if isinstance(val, torch.Tensor):
        val = val.item()
    return torch.nn.Parameter(torch.tensor(val, dtype=torch.float))


class BellMembFunc(torch.nn.Module):
    '''
        Generalised Bell membership function; defined by three parameters:
            a, the half-width (at the crossover point)
            b, controls the slope at the crossover point (which is -b/2a)
            c, the center point
    '''

    def __init__(self, a, b, c):
        super(BellMembFunc, self).__init__()
        self.register_parameter('a', _mk_param(a))
        self.register_parameter('b', _mk_param(b))
        self.register_parameter('c', _mk_param(c))
        self.b.register_hook(BellMembFunc.b_log_hook)

    @staticmethod
    def b_log_hook(grad):
        '''
            Possibility of a log(0) in the grad for b, giving a nan.
            Fix this by replacing any nan in the grad with ~0.
        '''
        grad[torch.isnan(grad)] = 1e-9
        return grad

    def forward(self, x):
        dist = torch.pow((x - self.c) / self.a, 2)
        return torch.reciprocal(1 + torch.pow(dist, self.b))

    def pretty(self):
        return 'BellMembFunc {} {} {}'.format(self.a, self.b, self.c)


class FuzzifyVariable(torch.nn.Module):
    '''
        Represents a single fuzzy variable, holds a list of its MFs.
        Forward pass will then fuzzify the input (value for each MF).
    '''

    def __init__(self, mfdefs):
        super(FuzzifyVariable, self).__init__()
        if isinstance(mfdefs, list):  # No MF names supplied
            mfnames = ['mf{}'.format(i) for i in range(len(mfdefs))]
            mfdefs = OrderedDict(zip(mfnames, mfdefs))
        self.mfdefs = torch.nn.ModuleDict(mfdefs)
        self.padding = 0

    @property
    def num_mfs(self):
        '''Return the actual number of MFs (ignoring any padding)'''
        return len(self.mfdefs)

    def members(self):
        '''
            Return an iterator over this variables's membership functions.
            Yields tuples of the form (mf-name, MembFunc-object)
        '''
        return self.mfdefs.items()

    def pad_to(self, new_size):
        '''
            Will pad result of forward-pass (with zeros) so it has new_size,
            i.e. as if it had new_size MFs.
        '''
        self.padding = new_size - len(self.mfdefs)

    def fuzzify(self, x):
        '''
            Yield a list of (mf-name, fuzzy values) for these input values.
        '''
        for mfname, mfdef in self.mfdefs.items():
            yvals = mfdef(x)
            yield (mfname, yvals)

    def forward(self, x):
        '''
            Return a tensor giving the membership value for each MF.
            x.shape: n_cases
            y.shape: n_cases * n_mfs
        '''
        y_pred = torch.cat([mf(x) for mf in self.mfdefs.values()], dim=1)
        if self.padding > 0:
            y_pred = torch.cat([y_pred,
                                torch.zeros(x.shape[0], self.padding)], dim=1)
        return y_pred


class FuzzifyLayer(torch.nn.Module):
    '''
        A list of fuzzy variables, representing the inputs to the FIS.
        Forward pass will fuzzify each variable individually.
        We pad the variables so they all seem to have the same number of MFs,
        as this allows us to put all results in the same tensor.
    '''

    def __init__(self, varmfs, varnames=None):
        # print('varmfs')
        # print(varmfs)
        super(FuzzifyLayer, self).__init__()
        if not varnames:
            self.varnames = ['x{}'.format(i) for i in range(len(varmfs))]
        else:
            self.varnames = list(varnames)
        maxmfs = max([var.num_mfs for var in varmfs])
        for var in varmfs:
            var.pad_to(maxmfs)
        self.varmfs = torch.nn.ModuleDict(zip(self.varnames, varmfs))

    @property
    def num_in(self):
        '''Return the number of input variables'''
        return len(self.varmfs)

    @property
    def max_mfs(self):
        ''' Return the max number of MFs in any variable'''
        return max([var.num_mfs for var in self.varmfs.values()])

    def __repr__(self):
        '''
            Print the variables, MFS and their parameters (for info only)
        '''
        r = ['Input variables']
        for varname, members in self.varmfs.items():
            r.append('Variable {}'.format(varname))
            for mfname, mfdef in members.mfdefs.items():
                r.append('- {}: {}({})'.format(mfname,
                                               mfdef.__class__.__name__,
                                               ', '.join(['{}={}'.format(n, p.item())
                                                          for n, p in mfdef.named_parameters()])))
        return '\n'.join(r)

    def forward(self, x):
        ''' Fuzzyify each variable's value using each of its corresponding mfs.
            x.shape = n_cases * n_in
            y.shape = n_cases * n_in * n_mfs
        '''
        assert x.shape[1] == self.num_in, \
            '{} is wrong no. of input values'.format(self.num_in)
        y_pred = torch.stack([var(x[:, i:i + 1])
                              for i, var in enumerate(self.varmfs.values())],
                             dim=1)
        return y_pred


class AntecedentLayer(torch.nn.Module):
    '''
        Form the 'rules' by taking all possible combinations of the MFs
        for each variable. Forward pass then calculates the fire-strengths.
    '''

    def __init__(self, varlist):
        super(AntecedentLayer, self).__init__()
        # Count the (actual) mfs for each variable:
        mf_count = [var.num_mfs for var in varlist]
        # Now make the MF indices for each rule:
        mf_indices = itertools.product(*[range(n) for n in mf_count])
        self.mf_indices = torch.tensor(list(mf_indices))
        # mf_indices.shape is n_rules * n_in

    def num_rules(self):
        return len(self.mf_indices)

    def extra_repr(self, varlist=None):
        if not varlist:
            return None
        row_ants = []
        mf_count = [len(fv.mfdefs) for fv in varlist.values()]
        for rule_idx in itertools.product(*[range(n) for n in mf_count]):
            thisrule = []
            for (varname, fv), i in zip(varlist.items(), rule_idx):
                thisrule.append('{} is {}'
                                .format(varname, list(fv.mfdefs.keys())[i]))
            row_ants.append(' and '.join(thisrule))
        return '\n'.join(row_ants)

    def forward(self, x):
        ''' Calculate the fire-strength for (the antecedent of) each rule
            x.shape = n_cases * n_in * n_mfs
            y.shape = n_cases * n_rules
        '''
        # Expand (repeat) the rule indices to equal the batch size:
        batch_indices = self.mf_indices.expand((x.shape[0], -1, -1))
        # Then use these indices to populate the rule-antecedents
        ants = torch.gather(x.transpose(1, 2), 1, batch_indices)
        # ants.shape is n_cases * n_rules * n_in
        # Last, take the AND (= product) for each rule-antecedent
        rules = torch.prod(ants, dim=2)
        return rules


class ConsequentLayer(torch.nn.Module):
    '''
        A simple linear layer to represent the TSK consequents.
        Hybrid learning, so use MSE (not BP) to adjust coefficients.
        Hence, coeffs are no longer parameters for backprop.
    '''

    def __init__(self, d_in, d_rule, d_out):
        super(ConsequentLayer, self).__init__()
        c_shape = torch.Size([d_rule, d_out, d_in + 1])
        self._coeff = torch.zeros(c_shape, dtype=dtype, requires_grad=True)

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self._coeff

    @coeff.setter
    def coeff(self, new_coeff):
        '''
            Record new coefficients for all the rules
            coeff: for each rule, for each output variable:
                   a coefficient for each input variable, plus a constant
        '''
        assert new_coeff.shape == self.coeff.shape, \
            'Coeff shape should be {}, but is actually {}' \
                .format(self.coeff.shape, new_coeff.shape)
        self._coeff = new_coeff

    def fit_coeff(self, x, weights, y_actual):
        '''
            Use LSE to solve for coeff: y_actual = coeff * (weighted)x
                  x.shape: n_cases * n_in
            weights.shape: n_cases * n_rules
            [ coeff.shape: n_rules * n_out * (n_in+1) ]
                  y.shape: n_cases * n_out
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Shape of weighted_x is n_cases * n_rules * (n_in+1)
        weighted_x = torch.einsum('bp, bq -> bpq', weights, x_plus)
        # Can't have value 0 for weights, or LSE won't work:
        weighted_x[weighted_x == 0] = 1e-12
        # Squash x and y down to 2D matrices for gels:
        weighted_x_2d = weighted_x.view(weighted_x.shape[0], -1)
        y_actual_2d = y_actual.view(y_actual.shape[0], -1)
        # Use gels to do LSE, then pick out the solution rows:
        try:
            # print(y_actual_2d.size(), weighted_x_2d.size())
            coeff_2d, _ = torch.lstsq(y_actual_2d, weighted_x_2d)
        except RuntimeError as e:
            print('Internal error in gels', e)
            print('Weights are:', weighted_x)
            raise e
        coeff_2d = coeff_2d[0:weighted_x_2d.shape[1]]
        # Reshape to 3D tensor: divide by rules, n_in+1, then swap last 2 dims
        self.coeff = coeff_2d.view(weights.shape[1], x.shape[1] + 1, -1) \
            .transpose(1, 2)
        # coeff dim is thus: n_rules * n_out * (n_in+1)

    def forward(self, x):
        '''
            Calculate: y = coeff * x + const   [NB: no weights yet]
                  x.shape: n_cases * n_in
              coeff.shape: n_rules * n_out * (n_in+1)
                  y.shape: n_cases * n_out * n_rules
        '''
        # Append 1 to each list of input vals, for the constant term:
        x_plus = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        # Need to switch dimansion for the multipy, then switch back:
        # print(type(self.coeff), type(x_plus.t()))
        y_pred = torch.matmul(self.coeff.double(), x_plus.t().double())  # 矩阵相乘
        return y_pred.transpose(0, 2)  # swaps cases and rules


class PlainConsequentLayer(ConsequentLayer):
    '''
        A linear layer to represent the TSK consequents.
        Not hybrid learning, so coefficients are backprop-learnable parameters.
    '''

    def __init__(self, *params):
        super(PlainConsequentLayer, self).__init__(*params)
        self.register_parameter('coefficients',
                                torch.nn.Parameter(self._coeff))

    @property
    def coeff(self):
        '''
            Record the (current) coefficients for all the rules
            coeff.shape: n_rules * n_out * (n_in+1)
        '''
        return self.coefficients

    def fit_coeff(self, x, weights, y_actual):
        '''
        '''
        assert False, \
            'Not hybrid learning: I\'m using BP to learn coefficients'


class WeightedSumLayer(torch.nn.Module):
    '''
        Sum the TSK for each outvar over rules, weighted by fire strengths.
        This could/should be layer 5 of the Anfis net.
        I don't actually use this class, since it's just one line of code.
    '''

    def __init__(self):
        super(WeightedSumLayer, self).__init__()

    def forward(self, weights, tsk):
        '''
            weights.shape: n_cases * n_rules
                tsk.shape: n_cases * n_out * n_rules
             y_pred.shape: n_cases * n_out
        '''
        # Add a dimension to weights to get the bmm to work:
        y_pred = torch.bmm(tsk, weights.unsqueeze(2))
        return y_pred.squeeze(2)


class RF_ANFISModel(torch.nn.Module):
    '''
        This is a container for the 5 layers of the ANFIS net.
        The forward pass maps inputs to outputs based on current settings,
        and then fit_coeff will adjust the TSK coeff using LSE.
    '''

    def __init__(self, **kwargs):
        super(RF_ANFISModel, self).__init__()
        if "sigma" not in kwargs.keys():
            self.sigma = 1
        else:
            self.sigma = kwargs["sigma"]
        # self.description = description
        self.invardefs = [
            ('x0', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ('x1', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
            ('x2', make_bell_mfs(3.33333, 2, [-10, -3.333333, 3.333333, 10])),
        ]
        self.outvarnames = ['y0']

        #self.hybrid = hybrid

        varnames = [v for v, _ in self.invardefs]
        mfdefs = [FuzzifyVariable(mfs) for _, mfs in self.invardefs]
        self.num_in = len(self.invardefs)
        self.num_rules = np.prod([len(mfs) for _, mfs in self.invardefs])

        # print(self.num_rules)
        #if self.hybrid:
        cl = ConsequentLayer(self.num_in, self.num_rules, self.num_out)
        #else:
        #    cl = PlainConsequentLayer(self.num_in, self.num_rules, self.num_out)

        self.layer = torch.nn.ModuleDict(OrderedDict([
            ('fuzzify', FuzzifyLayer(mfdefs, varnames)),
            ('rules', AntecedentLayer(mfdefs)),
            # normalisation layer is just implemented as a function.
            ('consequent', cl),
            # weighted-sum layer is just implemented as a function.
        ]))

    @property
    def num_out(self):
        return len(self.outvarnames)

    @property
    def coeff(self):
        return self.layer['consequent'].coeff

    @coeff.setter
    def coeff(self, new_coeff):
        self.layer['consequent'].coeff = new_coeff

    def x_filter(self, x, y, sigma):
        x_np = x.T
        y_np = y.T
        # print(x_np.shape,y_np.shape)
        # for i in x_np:
        corration = np.corrcoef(x_np, y_np)
        #print("corr:", corration)
        tobedelete = []
        for i in range(x_np.shape[0]):
            if np.abs(corration[i, x_np.shape[0]]) <= sigma:
                tobedelete.append(i)
        x_np = np.delete(x_np, tobedelete, axis=0)
        return x_np

    def fit_coeff(self, x, y_actual):
        '''
            Do a forward pass (to get weights), then fit to y_actual.
            Does nothing for a non-hybrid ANFIS, so we have same interface.
        '''
        #if self.hybrid:
        self(x)
        self.layer['consequent'].fit_coeff(x, self.weights, y_actual)

    def input_variables(self):
        '''
            Return an iterator over this system's input variables.
            Yields tuples of the form (var-name, FuzzifyVariable-object)
        '''
        return self.layer['fuzzify'].varmfs.items()

    def output_variables(self):
        '''
            Return an list of the names of the system's output variables.
        '''
        return self.outvarnames

    def extra_repr(self):
        rstr = []
        vardefs = self.layer['fuzzify'].varmfs
        rule_ants = self.layer['rules'].extra_repr(vardefs).split('\n')
        for i, crow in enumerate(self.layer['consequent'].coeff):
            rstr.append('Rule {:2d}: IF {}'.format(i, rule_ants[i]))
            rstr.append(' ' * 9 + 'THEN {}'.format(crow.tolist()))
        return '\n'.join(rstr)

    def forward(self, x):
        '''
            Forward pass: run x thru the five layers and return the y values.
            I save the outputs from each layer to an instance variable,
            as this might be useful for comprehension/debugging.
        '''
        self.fuzzified = self.layer['fuzzify'](x)
        self.raw_weights = self.layer['rules'](self.fuzzified)
        self.weights = F.normalize(self.raw_weights, p=1, dim=1)
        self.rule_tsk = self.layer['consequent'](x)
        # y_pred = self.layer['weighted_sum'](self.weights, self.rule_tsk)
        y_pred = torch.bmm(self.rule_tsk.double(), self.weights.unsqueeze(2).double())
        self.y_pred = y_pred.squeeze(2)
        return self.y_pred

    def fit(self, **kwargs):
        '''
            Train the given model using the given (x,y) data.
        '''
        epochs = 10
        show_plots = False
        self.trainX = kwargs["trainX"]
        #print("tX:", self.trainX.shape)
        self.trainY = kwargs["trainY"]
        self.x_filter(self.trainX, self.trainY, self.sigma)
        #print("tX1:", self.trainX.shape)
        data = self.trainX, self.trainY
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-4, momentum=0.99)
        criterion = torch.nn.MSELoss(reduction='sum')
        train_anfis_with(self, data, optimizer, criterion, epochs, show_plots)

    def predict(self, **kwargs):
        # Get the error rate for the whole batch:
        # 处理test,将test从numpy二元组合并为dataLoader
        assert "predictX" in kwargs.keys()
        '''if len(kwargs.get("predictX"))==2:
            tensor_X = torch.from_numpy(kwargs.get("predictX")[0])
            tensor_y = torch.from_numpy(kwargs.get("predictX")[1])'''

        tensor_X = torch.from_numpy(kwargs.get("predictX"))
        tensor_y = torch.from_numpy(kwargs.get("predictX")[:, 0])
        ds = TensorDataset(tensor_X, tensor_y)
        test = DataLoader(ds, batch_size=16, shuffle=True)

        y_pred = np.array([[0]])
        for batch_x, batch_y in test:
            y_pred_batch = self(batch_x)
            # print(y_pred_batch.detach().numpy().shape)
            y_pred = np.concatenate((y_pred, y_pred_batch.detach().numpy()))
        y_pred = torch.from_numpy(y_pred[1:, :])
        y_pred = y_pred.numpy()
        return y_pred

    def fitForUI(self, **kwargs):
        """ 返回结果到前端
        :return:
        """
        assert "trainX" in kwargs.keys()
        assert "trainY" in kwargs.keys()
        trainX = kwargs["trainX"]
        trainY = kwargs["trainY"]
        self.fit(trainX=trainX, trainY=trainY)
        # 返回结果为字典形式
        # excludeFeatures, coefs = self.fit(**kwargs)
        returnDic = {
            str(None): str(None),
            str(None): str(None)
        }
        return returnDic

    def testForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        returnDic = {
            "mean_squared_error": None,
            "mean_absolute_error": None
        }
        testX = kwargs.get("testX")
        predictResult = self.predict(predictX=testX)
        testY = kwargs.get("testY")
        mse = mean_squared_error(predictResult, testY)
        mae = mean_absolute_error(predictResult, testY)
        returnDic["mean_absolute_error"] = str(mae)
        returnDic["mean_squared_error"] = str(mse)
        return returnDic

    def predictForUI(self, **kwargs):
        """
        :param kwargs:
        :return: 字典形式结果
        """
        returnDic = {
            "预测结果": None
        }
        predictResult = self.predict(**kwargs)
        returnDic["预测结果"] = str(predictResult)
        return returnDic


if __name__ == '__main__' and False:
    model = RF_ANFISModel()
    data = ANFISDataLoader()
    train_data = data.loadTrainData()
    # print(train_data[0].shape, train_data[1].shape)
    test_data = data.loadTestData()
    # train_anfis(model, train_data, 20)
    model.fit(train_data)
    # print(type(train_data))
    predict_y = model.predict(test_data)

if __name__ == "__main__" and True:

    from sklearn.metrics import mean_squared_error

    model = RF_ANFISModel(sigma=0.4)

    data = ANFISDataLoader()
    train_data = data.loadTrainData(train_path='../data/rfanfis/RFANFIS_TRAIN_DATA.xlsx')
    print("train_data", train_data)
    test_data = data.loadTestData(test_path='../data/rfanfis/RFANFIS_TEST_DATA.xlsx')
    predict_data = data.loadPredictData(predict_path='../data/rfanfis/RFANFIS_PREDICT_DATA.xlsx')
    model.fitForUI(train_data=train_data)
    predictY = model.predict(predictX=predict_data)
    predictYForUI = model.predictForUI(predictX=predict_data)
    predictTrainY = model.predict(predictX=predict_data)
    # print(model.model.coef_)
